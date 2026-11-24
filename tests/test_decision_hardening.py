"""决策层回归：RSI 门控/评测的重复、冻结回放、时点与判官覆盖；模拟盘的未来日期、停牌与回撤口径。"""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.rsi.evaluator import JudgeOut, evaluate_s1, evaluate_s2, evaluate_s3, frozen_replay, search_f
from advisor.rsi.gates import cost_gate, noise_gate, overfit_gate, protocol_gate
from advisor.rsi.issues import cached_results, usable
from advisor.rsi.models import Issue, Run
from advisor.rsi.runner import RSIRunner
from advisor.services import Services
from advisor.strategy import PaperError, Rule
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of
from tests.test_phase4 import spec
from tests.test_rsi import seed_dossiers

AS_OF = "2026-10-16"


def _run(scenario="S2", kind="routing", **kw):
    now = datetime.now(timezone.utc)
    base = dict(run_id="r1", scenario=scenario, kind=kind, issue_ids=[], base_version=1,
                content={}, diff={}, change_reason="测试改动原因不少于十个字", direction="d", created_at=now, updated_at=now)
    return Run(**(base | kw))


def _slices(means: dict[str, dict], runs_n: int = 3, repeats: int = 3, primary="evolution"):
    return {"primary": primary, "repeats": repeats,
            "arms": {arm: {"slices": {k: {"runs": [v["mean"]] * runs_n, "mean": v["mean"],
                                          "spread": v.get("spread", 0.0)} for k, v in s.items()}}
                     for arm, s in means.items()}}


# ---------- RSI 门控 ----------

def test_noise_gate_requires_three_real_samples(repo):
    from advisor.rsi import Ledger
    ev = _slices({"base": {"evolution": {"mean": 0.0}}, "cand": {"evolution": {"mean": 1.0}}}, runs_n=1, repeats=4)
    gate = noise_gate(_run(), ev, Ledger(repo.db))
    assert not gate.passed and "实际重复 1 次" in gate.detail


def test_cost_gate_base_zero(repo):
    from advisor.rsi import Ledger
    ev = _slices({"base": {"evolution": {"mean": 0.0}}, "cand": {"evolution": {"mean": 0.01}}})
    ev["arms"]["base"]["tokens"], ev["arms"]["cand"]["tokens"] = 100, 600
    gate = cost_gate(_run(), ev, Ledger(repo.db))
    assert not gate.passed and "1000000000" not in gate.detail and "相对提升无意义" in gate.detail


def test_overfit_gate_allows_grid_rounding(repo, monkeypatch):
    from advisor.rsi import Ledger
    ledger = Ledger(repo.db)
    monkeypatch.setattr(ledger, "released_count", lambda kind: 2)
    run = _run(diff={"params": {"temperature": [0.75, 0.917]}})
    gate = overfit_gate(run, {}, ledger)
    assert gate.passed, gate.detail                      # 0.167 的网格点不应被 1/6 的精确上限误拒


def test_protocol_gate_fails_when_replay_partially_loaded(repo):
    from advisor.rsi import Ledger
    from advisor.rsi.evaluator import _h
    ledger = Ledger(repo.db)
    frozen_ids = ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    repo.db.rsi_protocols.insert_one({"scenario": "S3", "replay_ids": frozen_ids, "hash": _h(frozen_ids)})
    ev = _slices({"base": {"evolution": {"mean": 1.0}}, "cand": {"evolution": {"mean": 1.0}}})
    ev["protocol"] = {"replay_hash": _h(frozen_ids), "replay_loaded": 0, "replay_frozen": 2}
    gate = protocol_gate(_run("S3", "search"), ev, ledger)
    assert not gate.passed and "只加载到 0 项" in gate.detail


# ---------- RSI 评测 ----------

def test_s2_ignores_failed_eval_runs(repo, policy):
    later = datetime.now(timezone.utc) + timedelta(days=1)
    for i in range(6):                                          # 失败文档更新且超过 limit=5：不筛状态就会把可评分运行挤出窗口
        repo.db.eval_runs.insert_one({"run_id": f"rt_fail{i}", "type": "routing", "status": "failed",
                                      "error": "x", "created_at": later})
    repo.db.eval_runs.insert_one({"run_id": "rt_ok", "type": "routing", "status": "done", "created_at": datetime.now(timezone.utc),
                                  "rows": [{"question": "q1", "label": "standard", "route": "standard", "source": "model",
                                            "probs": {"quick": 0.05, "standard": 0.5, "deep": 0.45},
                                            "stakes": "mid", "uncertainty": "low", "intent": "research"}]})
    from advisor.rsi import Ledger
    deps = SimpleNamespace(repo=repo, policy=policy)
    ev = evaluate_s2(deps, Ledger(repo.db), _run(), {"abstain_below": 0.55}, {"abstain_below": 0.5})
    assert ev["repeats"] == 1                                   # 失败的评测不算一次重复
    assert len(ev["arms"]["base"]["slices"]["replay"]["runs"]) == 1
    assert not noise_gate(_run(), ev, Ledger(repo.db)).passed


def test_frozen_replay_rejects_empty(repo):
    from advisor.rsi import Ledger
    with pytest.raises(ValueError, match="没有可冻结"):
        frozen_replay(Ledger(repo.db), "S3", [])


def test_s3_protocol_fails_when_frozen_ids_evaporated(repo, policy):
    from advisor.rsi import Ledger
    from advisor.rsi.evaluator import _h
    repo.db.cache.insert_one({"key": "ds:search:2026-10-16:10:示例搜索 after:2026-06-18 before:2026-10-17",
                              "value": [{"url": "https://www.cninfo.com.cn/a", "authority": 4, "rerank_score": 2.0,
                                         "date": "2026-10-15T00:00:00+00:00"}]})
    repo.db.rsi_protocols.insert_one({"scenario": "S3", "replay_ids": ["ffffffffffff"], "hash": _h(["ffffffffffff"])})
    ledger = Ledger(repo.db)
    deps = SimpleNamespace(repo=repo, policy=policy)
    cfg = {"rerank_min": 1.5, "authority_b_min": 3, "lookback_days": 120}
    ev = evaluate_s3(deps, ledger, _run("S3", "search"), cfg, dict(cfg, rerank_min=1.6))
    gate = protocol_gate(_run("S3", "search"), ev, ledger)
    assert not gate.passed and "只加载到 0 项" in gate.detail


def test_s3_search_f_drops_future_results(policy):
    searches = [{"id": "a", "as_of": AS_OF, "query": "q",
                 "raw": [{"url": "https://www.cninfo.com.cn/future", "authority": 5, "rerank_score": 2.0,
                          "date": "2026-10-20T00:00:00+00:00"}]}]
    out = search_f(searches, {"rerank_min": 1.5, "authority_b_min": 3}, policy)
    assert out["kept"] == 0 and out["f"] == 0.0


def test_usable_drops_future_results(repo):
    future = {"url": "https://www.cninfo.com.cn/future", "authority": 5, "rerank_score": 2.0,
              "date": "2026-10-20T00:00:00+00:00", "_cache_as_of": AS_OF}
    assert not usable(future, 3, ("cninfo.com.cn",), ())
    repo.db.cache.insert_one({"key": f"ds:search:{AS_OF}:10:示例搜索 after:2026-06-18",
                              "value": [{k: v for k, v in future.items() if k != "_cache_as_of"}]})
    assert cached_results(repo)["示例搜索"] == []


def test_s1_missing_verdicts_invalidates_eval(repo):
    from advisor.rsi import Ledger
    seed_dossiers(repo)
    ledger = Ledger(repo.db)
    issue = Issue(id="is1", scenario="S1", kind="challenge_upheld", case="指标变化直接归因，未排除替代解释",
                  evidence=["ch1"], hypothesis="研究员把指标变化直接归因到单一原因，没有先排除其他解释",
                  source={"dossier": "600001", "version": 1, "claim_id": "fundamental-c1", "ticker": "600001"},
                  fingerprint="fp1", created_at=datetime.now(timezone.utc))
    ledger.add_issue(issue)

    class _H:
        budget = SimpleNamespace(used={"tokens": 0})

        def llm(self, role, system, user, schema, purpose=None):
            self.budget.used["tokens"] += 1
            return JudgeOut(items=[])                 # 故意漏判所有条目

    deps = SimpleNamespace(repo=repo, harness=lambda *a, **k: _H())
    with pytest.raises(ValueError, match="漏判"):
        evaluate_s1(deps, ledger, _run("S1", "lessons", issue_ids=["is1"]), [], [])


def test_top_up_routing_counts_only_scorable_runs(repo, monkeypatch):
    import advisor.evals.routing as routing_mod
    for i in range(3):
        repo.db.eval_runs.insert_one({"run_id": f"rt_fail{i}", "type": "routing", "status": "failed"})
    repo.db.eval_runs.insert_one({"run_id": "rt_ok", "type": "routing", "status": "done", "rows": [{"question": "q"}]})
    calls = []
    monkeypatch.setattr(routing_mod, "run_routing_eval", lambda deps: calls.append(1))
    stub = SimpleNamespace(deps=SimpleNamespace(repo=repo))
    RSIRunner._top_up_routing(stub, lambda m: None)
    assert len(calls) == 2                                # 3 次目标 - 1 次可评分 = 补跑 2 次


# ---------- 策略卡与模拟盘 ----------

def test_rule_treats_drawdown_as_magnitude():
    assert Rule(field="mdd_250d", op=">=", value=20).check({"mdd_250d": Decimal("-60.00")}) is True
    assert Rule(field="mdd_250d", op="<=", value=20).check({"mdd_250d": Decimal("-60.00")}) is False


class _HaltedSource(FakeSource):
    """10-20 之后没有新行情（模拟停牌/周末的数据源行为）。"""

    def daily(self, code, start, end):
        return super().daily(code, start, min(end, date(2026, 10, 20)))


@pytest.fixture
def halted_services(repo, policy):
    deps = build_deps(repo, FakeLLM(), policy, source=_HaltedSource(), web_tools=FakeWebTools())
    svc = ResearchService(deps, InMemorySaver())
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    return Services(deps, svc)


def test_paper_rejects_future_day(halted_services):
    book = halted_services.strategy
    v = book.create(spec(), "owner", "未来日期不能推进模拟盘")
    v = book.start_paper(v.card_id, 1, "owner", start=date(2026, 10, 19))
    with pytest.raises(PaperError, match="未来"):
        book.paper.step(v, date.today() + timedelta(days=1))


def test_paper_skips_non_trading_day(halted_services):
    book = halted_services.strategy
    v = book.create(spec(), "owner", "停牌日没有当日报价就不生成净值点")
    v = book.start_paper(v.card_id, 1, "owner", start=date(2026, 10, 19))
    book.paper.step(v, date(2026, 10, 19))
    book.paper.step(v, date(2026, 10, 20))
    with pytest.raises(PaperError, match="没有当日行情"):
        book.paper.step(v, date(2026, 10, 21))
    assert [p.date for p in book.store.navs_of(v.card_id, v.version)] == [date(2026, 10, 19), date(2026, 10, 20)]
    assert not [t for t in book.store.list_trades(v.card_id, v.version) if t.date == date(2026, 10, 21)]
