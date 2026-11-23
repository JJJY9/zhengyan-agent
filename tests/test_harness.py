from datetime import date

import pytest
from pydantic import BaseModel

from advisor.contracts.skill import Skill, SkillSpec
from advisor.harness import Budget, BudgetExceeded, Harness, HarnessDenied, RunContext, SkillFailed
from advisor.skills.registry import SkillRegistry


class In(BaseModel):
    query: str = "x"
    as_of: date | None = None


class Out(BaseModel):
    n: int


def make(policy, skills, route="standard", limits=None, as_of=date(2026, 10, 18)):
    audit: list[dict] = []
    budget = Budget(limits=limits or {"search_calls": 2, "tokens": 100, "wall_clock_s": 60})
    h = Harness(policy, SkillRegistry(skills), audit.append, budget, RunContext("t", route, as_of))
    return h, audit, budget


def skill(name, run=lambda i: Out(n=1), **kw):
    timeout_s = kw.pop("timeout_s", 2)
    return Skill(spec=SkillSpec(name=name, version="1", category="data", permission="read", description="",
                                input_model=In, output_model=Out, timeout_s=timeout_s, **kw), run=run)


def test_unregistered_and_not_whitelisted_denied(policy):
    h, audit, _ = make(policy, [skill("market.daily"), skill("secret.tool")])
    with pytest.raises(HarnessDenied):
        h.call("nope.tool")
    with pytest.raises(HarnessDenied, match="无权"):
        h.call("secret.tool")
    assert h.call("market.daily").n == 1
    assert [a["status"] for a in audit] == ["HarnessDenied", "HarnessDenied", "ok"]


def test_quick_route_cannot_fetch(policy):
    h, _, _ = make(policy, [skill("web.fetch")], route="quick")
    with pytest.raises(HarnessDenied):
        h.call("web.fetch")


def test_action_skills_denied_by_approval_gate(policy, monkeypatch):
    monkeypatch.setattr(policy, "is_allowed", lambda r, s: True)
    h, _, _ = make(policy, [skill("action.place_order")])
    with pytest.raises(HarnessDenied, match="审批门"):
        h.call("action.place_order")


def test_future_as_of_denied(policy):
    h, _, _ = make(policy, [skill("market.daily")])
    with pytest.raises(HarnessDenied, match="晚于研究时点"):
        h.call("market.daily", as_of=date(2026, 12, 1))


def test_egress_blocks_sensitive_query(policy):
    h, _, _ = make(policy, [skill("web.search", egress_fields=("query",), budget_kind="search_calls")])
    with pytest.raises(HarnessDenied, match="敏感"):
        h.call("web.search", query="我的持仓 账户 6222020200112233445")
    assert h.call("web.search", query="示例酒业 业绩").n == 1


def test_budget_and_idempotency(policy):
    calls = []
    s = skill("web.search", run=lambda i: calls.append(i) or Out(n=len(calls)), egress_fields=("query",), budget_kind="search_calls")
    h, _, budget = make(policy, [s])
    h.call("web.search", query="a")
    h.call("web.search", query="a")          # 幂等：命中缓存，不重复执行、不重复计费
    h.call("web.search", query="b")
    assert len(calls) == 2 and budget.used["search_calls"] == 2
    with pytest.raises(BudgetExceeded):
        h.call("web.search", query="c")


def test_idempotency_is_task_scoped_across_nodes(deps):
    """同一任务的不同节点（各自新建 Harness）共用幂等缓存；不同任务互不影响。"""
    from datetime import date as _d
    calls = []
    deps.registry.register(skill("x.echo", run=lambda i: calls.append(i) or Out(n=len(calls))))
    deps.policy.routes["standard"] = deps.policy.routes["standard"].__class__(
        **{**deps.policy.routes["standard"].__dict__, "allowed_skills": deps.policy.routes["standard"].allowed_skills + ("x.*",)})
    b = lambda: Budget(limits={"wall_clock_s": 60})   # noqa: E731
    h1 = deps.harness("t_same", "standard", _d(2026, 10, 18), b(), node="collect", task_scoped_cache=True)
    h2 = deps.harness("t_same", "standard", _d(2026, 10, 18), b(), node="search", task_scoped_cache=True)
    h3 = deps.harness("t_other", "standard", _d(2026, 10, 18), b(), node="collect", task_scoped_cache=True)
    assert h1.call("x.echo", query="q").n == 1
    assert h2.call("x.echo", query="q").n == 1          # 同任务另一个节点：命中缓存，不重复执行
    assert h3.call("x.echo", query="q").n == 2          # 另一个任务：重新执行
    assert len(calls) == 2


def test_retry_only_retryable(policy):
    attempts = []

    def flaky(i):
        attempts.append(1)
        if len(attempts) < 2:
            raise SkillFailed("429", retryable=True)
        return Out(n=7)

    h, _, _ = make(policy, [skill("market.daily", run=flaky)])
    assert h.call("market.daily").n == 7 and len(attempts) == 2

    def broken(i):
        attempts.append(1)
        raise SkillFailed("402 余额不足", retryable=False)

    attempts.clear()
    h2, _, _ = make(policy, [skill("market.daily", run=broken)])
    with pytest.raises(SkillFailed):
        h2.call("market.daily")
    assert len(attempts) == 1


def test_wall_clock_deadline(policy):
    h, _, budget = make(policy, [skill("market.daily")], limits={"wall_clock_s": 10})
    budget.started_at -= 100
    with pytest.raises(BudgetExceeded, match="时限"):
        h.call("market.daily")


def test_safety_check_failure_blocks(policy, monkeypatch):
    h, _, _ = make(policy, [skill("web.search", egress_fields=("query",))])
    monkeypatch.setattr(h.egress, "check_outbound", lambda t: 1 / 0)
    with pytest.raises(HarnessDenied, match="默认阻断"):
        h.call("web.search", query="abc")


def test_llm_redacts_and_charges(policy):
    seen = {}

    class LLM:
        def json(self, role, system, user, schema, max_tokens=None):
            seen["user"] = user
            return Out(n=1), 40

    h, audit, budget = make(policy, [])
    h.llm_client = LLM()
    h.llm("draft", "s", "我的持仓 30万元 怎么办", Out, purpose="x")
    assert "30万" not in seen["user"] and budget.used["tokens"] == 40
    assert audit[-1]["kind"] == "llm"


def test_decision_call_is_redacted_and_audited(policy):
    from tests.fakes import FakeDecision
    h, audit, _ = make(policy, [])
    h.decision_client = FakeDecision()
    out = h.decide("我的账户有 50 万元，茅台估值合理吗", {"route": ("which?", {"quick": None, "standard": None, "deep": None})},
                   purpose="route_classify_small")
    assert set(out["route"]) == {"quick", "standard", "deep"}
    assert "我的账户" not in h.decision_client.calls[0] and "[已脱敏]" in h.decision_client.calls[0]
    assert audit[-1]["kind"] == "decision" and audit[-1]["status"] == "ok" and audit[-1]["model"] == "fake-kev"


def test_egress_phone_pattern_ignores_large_amounts(policy):
    from advisor.harness.egress import EgressGuard
    g = EgressGuard(policy.egress["sensitive_patterns"], policy.egress.get("trade_intent_patterns", []))
    assert g.find_sensitive("营业收入 15000000000 元") == []                 # 150 亿不是手机号
    text = "营收15000000000元，同比+3%"
    assert g.redact(text) == text
    assert g.find_sensitive("联系电话 13812345678") == ["13812345678"]      # 真手机号仍然命中
    assert "[已脱敏]" in g.redact("电话13812345678")


def test_cache_collection_has_ttl_index(repo):
    info = repo.db.cache.index_information()
    assert any("expires_at" in k and v.get("expireAfterSeconds") == 0 for k, v in info.items())


def test_llm_completion_capped_by_remaining_tokens(policy):
    seen = {}

    class LLM:
        def json(self, role, system, user, schema, max_tokens=None):
            seen["max_tokens"] = max_tokens
            return Out(n=1), 5

    h, _, _ = make(policy, [])                     # 默认 tokens 上限 100
    h.llm_client = LLM()
    h.llm("draft", "s", "u", Out, purpose="x")
    assert seen["max_tokens"] == 100                # 剩余不足：按剩余额度封顶
    h2, _, _ = make(policy, [], limits={"tokens": 20000, "wall_clock_s": 60})
    h2.llm_client = LLM()
    h2.llm("draft", "s", "u", Out, purpose="x")
    assert seen["max_tokens"] == 8192               # 预算充足：封顶 8192，不放大模型默认输出长度


def test_timeouts_do_not_poison_call_capacity(policy, monkeypatch):
    """超时只放弃等待：单个技能跑飞不能占满共享容量，健康技能必须照常秒回。"""
    import threading

    import advisor.harness.executor as ex
    monkeypatch.setattr(ex, "MAX_RETRIES", 0)              # 单次尝试，省去退避等待

    def hang(i):
        threading.Event().wait(10)                        # 永不按时返回（守护线程，测试结束即回收）
        return Out(n=1)

    h, _, _ = make(policy, [skill("market.daily", run=hang, timeout_s=0.02)],
                   limits={"tokens": 10, "wall_clock_s": 60})
    for _ in range(17):                                    # 超过旧实现 16 线程池的容量
        with pytest.raises(SkillFailed, match="超时"):
            h.call("market.daily")
    h2, _, _ = make(policy, [skill("financials.statements")])
    assert h2.call("financials.statements").n == 1        # 旧实现：健康调用排队到超时
    workers = [t for t in threading.enumerate() if t.name == "harness-skill"]
    assert workers and all(t.daemon for t in workers)


def test_retries_are_charged_per_attempt(policy, monkeypatch):
    """每次真实出站调用都扣预算：重试不是免费的，预算耗尽即停止重试。"""
    import advisor.harness.executor as ex
    monkeypatch.setattr(ex.time, "sleep", lambda *_: None)     # 去退避

    calls = {"n": 0}

    def flaky(i):
        calls["n"] += 1
        raise ConnectionError("down")

    s = skill("web.search", run=flaky, budget_kind="search_calls", retryable=(ConnectionError,))
    h, _, budget = make(policy, [s], limits={"search_calls": 2, "tokens": 10, "wall_clock_s": 60})
    with pytest.raises(BudgetExceeded):
        h.call("web.search", query="a")
    assert calls["n"] == 2 and budget.used["search_calls"] == 2


def test_task_cache_single_flight(policy):
    """并行专家用相同参数调用同一技能：只执行一次，其余等待并复用结果。"""
    import threading
    import time as clock

    calls = {"n": 0}
    gate = threading.Barrier(5)

    def slow(i):
        calls["n"] += 1
        clock.sleep(0.2)
        return Out(n=1)

    cache, flight = {}, {}
    results, errors = [], []

    def worker():
        b = Budget(limits={"search_calls": 5, "tokens": 10, "wall_clock_s": 60})
        h = Harness(policy, SkillRegistry([skill("market.daily", run=slow)]), lambda r: None, b,
                    RunContext("t", "standard", date(2026, 10, 18)), cache=cache, flight=flight)
        try:
            gate.wait(timeout=5)
            results.append(h.call("market.daily").n)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors and results == [1] * 5
    assert calls["n"] == 1
