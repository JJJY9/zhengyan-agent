"""R88a 回归：缓存键格式稳定与去重、日期形 scope、side admission 仅前台。

历史用户键沿用原格式（S3 冻结回放 id 稳定）；旧键与 scope 键并存时统计只算一次；
日期形 namespace 不会被误判为用户；admission='side' 不允许后台提交（会绕过名额与停机收尾）。"""
import hashlib
import json
from datetime import date, datetime, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.harness import Budget
from advisor.rsi.evaluator import _cached_searches
from advisor.rsi.issues import _search_key_parts, cached_results
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of
from tests.test_phase2 import make_service

AS_OF = "2026-10-16"
RAW = [{"title": f"t{i}", "url": f"https://www.stcn.com/a/{i}.html", "authority": 4,
        "rerank_score": 2.0, "date": "2026-08-01T00:00:00+00:00"} for i in range(8)]


def _put(repo, key, raw=RAW):
    repo.db.cache.insert_one({"key": key, "value": raw,
                              "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc)})


def test_user_writes_keep_legacy_key_format(repo, policy):
    """用户搜索沿用历史键格式：S3 冻结回放 id 不因升级而失效。"""
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    h = deps.harness("t_fmt", "standard", date.fromisoformat(AS_OF),
                     Budget(limits=policy.route("standard").budget))
    h.call("web.search", query="示例酒业 业绩 fmt", as_of=date.fromisoformat(AS_OF), count=10, lookback_days=120)
    keys = [d["key"] for d in repo.db.cache.find({"key": {"$regex": "^ds:search:"}})]
    assert keys
    assert all(k.startswith(f"ds:search:{AS_OF}:") for k in keys)   # 无 scope 前缀
    assert all(_search_key_parts(k)[0] == "user" for k in keys)


def test_legacy_and_scoped_duplicates_counted_once(repo):
    """同一 (as_of, count, query) 的旧键与 scope 键并存：统计不翻倍，回放 id 与历史公式一致。"""
    q = "示例酒业 业绩 dup"
    _put(repo, f"ds:search:{AS_OF}:8:{q}")
    _put(repo, f"ds:search:user:{AS_OF}:8:{q}")                     # R86 窗口期写下的同内容副本
    assert len(cached_results(repo)[q]) == 8                        # 不是 16：阈值/分母不被放大
    searches = _cached_searches(repo)
    assert len(searches) == 1
    expect = hashlib.sha256(json.dumps([f"ds:search:{AS_OF}:8:{q}"], ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:12]
    assert searches[0]["id"] == expect                              # 与升级前冻结签名同源公式


def test_date_shaped_scope_not_treated_as_user(repo):
    """日期形 namespace 的缓存键按位置解析为 scope，不落历史用户分支。"""
    q = "示例酒业 业绩 date-scope"
    _put(repo, f"ds:search:2026-10-18:{AS_OF}:10:{q}")
    parsed = _search_key_parts(f"ds:search:2026-10-18:{AS_OF}:10:{q}")
    assert parsed == ("2026-10-18", AS_OF, "10", q)
    assert cached_results(repo) == {} and _cached_searches(repo) == []


def test_side_admission_rejects_background(repo, policy):
    """admission='side' 只支持前台：后台提交会绕过交互名额与停机收尾，直接拒绝。"""
    svc = make_service(repo, policy)
    req = ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="quick")
    with pytest.raises(ValueError, match="前台"):
        svc.start(req, background=True, admission="side")
    assert not repo.list_tasks()                                    # 拒绝发生在建任务之前
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    svc2 = ResearchService(deps, InMemorySaver())
    svc2.deps.repo.create_task("t_wait", {"question": "q"})
    repo.db.tasks.update_one({"task_id": "t_wait"}, {"$set": {"status": "waiting_approval"}})
    with pytest.raises(ValueError, match="前台"):
        svc2.resume("t_wait", approve=True, background=True, admission="side")


def test_deep_send_carries_namespace(repo, policy, monkeypatch):
    """深研 Send 载荷必须显式带 namespace：旧检查点兜底不能替代新任务的显式传递。"""
    from advisor.evals.compare import run_compare
    from advisor.experts import agent as agent_mod

    seen: list[str | None] = []
    real = agent_mod.run_expert

    def spy(deps, p):
        seen.append(p.get("namespace"))
        return real(deps, p)

    monkeypatch.setattr(agent_mod, "run_expert", spy)
    svc = make_service(repo, policy)
    run_compare(svc, [{"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())}],
                ["deep"], run_id="ev_ns_payload")
    assert seen and all(ns and ns.startswith("eval-") for ns in seen)
