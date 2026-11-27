"""R85a/R86a 命名空间隔离回归：watch 扫描/个人情境/策略卡拒绝评测命名空间 key、
搜索缓存 provenance（评测缓存不驱动 RSI）、评测用例不占交互研究名额。"""
from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from advisor.api.app import create_app
from advisor.contracts import ResearchRequest
from advisor.evals.compare import run_compare
from advisor.graph.deps import build_deps
from advisor.graph.runner import PoolSaturated, ResearchService
from advisor.harness import Budget
from advisor.rsi.evaluator import _cached_searches
from advisor.rsi.issues import _search_key_parts, cached_results
from advisor.services import Services
from advisor.strategy import StrategySpec, StrategyVersion
from advisor.strategy.models import Rule
from advisor.watch.models import Alert
from advisor.watch.scanner import WatchError
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of
from tests.test_phase2 import make_service

CASE = {"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())}


def _alert_services(repo, policy):
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    services.watch.store.subscribe(CODE, "示例酒业", 10)
    services.watch.store.add_alert(Alert(id="al_u", ticker=CODE, name="示例酒业", dossier_key=CODE,
                                         dossier_version=1, fingerprint="fp_u", events=[], affected=[],
                                         created_at=datetime.now(timezone.utc)))
    return svc, services


def test_watch_scan_rejects_namespace_key(repo, policy):
    svc, services = _alert_services(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="quick"),
              namespace="eval-x", background=False)          # 评测档案真实存在，拒绝不能靠「查无此档」
    ns = f"eval-x:{CODE}"
    assert repo.current_dossier(ns) is not None
    with TestClient(create_app(deps=svc.deps, checkpointer=InMemorySaver())) as c:
        r = c.post("/api/watch/scan/eval-x%3A" + CODE)
        assert r.status_code == 422                           # 路径参数与订阅一样限定 6 位代码
    with pytest.raises(WatchError, match="命名空间"):
        services.watch.scan(ns)                               # 领域层同样拒绝


def test_namespace_alert_hidden_and_supersede_guarded(repo, policy):
    svc, services = _alert_services(repo, policy)
    store = services.watch.store
    store.add_alert(Alert(id="al_eval", ticker=CODE, name="评测", dossier_key=f"eval-x:{CODE}",
                          dossier_version=99, fingerprint="fp_e", events=[], affected=[],
                          created_at=datetime.now(timezone.utc)))
    with TestClient(create_app(deps=svc.deps, checkpointer=InMemorySaver())) as c:
        ids = {a["id"] for a in c.get("/api/watch").json()["alerts"]}
        assert "al_eval" not in ids and "al_u" in ids         # 评测提醒不进用户列表


def test_personal_fit_rejects_namespace_key(repo, policy):
    svc, services = _alert_services(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
              background=False)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="quick"),
              namespace="eval-x", background=False)          # 评测档案真实存在：隔离不靠「查无此档」
    assert repo.current_dossier(f"eval-x:{CODE}") is not None
    with TestClient(create_app(deps=svc.deps, checkpointer=InMemorySaver())) as c:
        assert c.get(f"/api/profile/fit/eval-x%3A{CODE}").status_code == 404
        assert c.get(f"/api/profile/fit/{CODE}").status_code == 200


def test_strategy_universe_rejects_namespace_key(repo, policy):
    from advisor.services import Services
    from advisor.strategy import StrategyError

    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    bad = StrategySpec(name="隔离测试", thesis="评测档案不能进入用户策略卡的候选股票列表",
                       universe=[f"eval-x:{CODE}"], entry=[Rule(field="status", op="==", value="committed")])
    with pytest.raises(StrategyError, match="6 位"):
        services.strategy.create(bad, "owner", "评测档案不能进入用户策略卡的候选股票列表")
    # 历史脏数据：读库不校验，取档时过滤命名空间 key，避免整页 500
    card = StrategyVersion(**{**bad.model_dump(), "card_id": "SC-X", "version": 1, "change_reason": "历史脏数据读取容错测试用例",
                              "sha256": "deadbeef", "created_by": "owner", "created_at": datetime.now(timezone.utc)})
    assert services.strategy.paper.universe(card) == []


def test_search_cache_provenance_and_rsi_scope(repo, policy):
    """评测命名空间下的搜索写进独立 scope；RSI 只读用户/历史缓存。"""
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    as_of_day = date(2026, 10, 18)

    def run_search(namespace: str, tag: str):
        h = deps.harness(f"t_{tag}", "standard", as_of_day,
                         Budget(limits=policy.route("standard").budget), namespace=namespace)
        h.call("web.search", query=f"示例酒业 业绩 {tag}", as_of=as_of_day, count=10, lookback_days=120)

    run_search("eval-ev_x-standard", "evalcase")
    keys = [d["key"] for d in repo.db.cache.find({"key": {"$regex": "^ds:search:"}})]
    assert keys and all(_search_key_parts(k)[0] == "eval-ev_x-standard" for k in keys[:3])
    assert cached_results(repo) == {}                         # 评测搜索不进用户坏案例
    assert _cached_searches(repo) == []                       # 也不冻结进 S3 回放

    run_search("", "usercase")
    user = cached_results(repo)
    assert any("示例酒业 业绩 usercase" in q for q in user)  # 用户搜索可见
    assert not any("evalcase" in q for q in user)             # 评测搜索仍被排除
    assert any("usercase" in s["query"] for s in _cached_searches(repo))


def test_legacy_cache_key_treated_as_user(repo):
    repo.db.cache.insert_one({"key": f"ds:search:{as_of()}:10:示例搜索 after:2026-06-18 before:2026-10-17",
                              "value": [{"title": "t", "url": "https://www.stcn.com/a/1.html",
                                         "date": "2026-08-16T08:00:00+00:00"}]})
    assert _search_key_parts(f"ds:search:{as_of()}:10:q")[0] == "user"
    assert cached_results(repo)                              # 历史键按用户命名空间兼容读取


def test_eval_case_does_not_consume_interactive_admission(repo, policy):
    svc = make_service(repo, policy)
    svc.max_pending = 1
    svc._acquire_or_raise()                                   # 交互名额已满
    try:
        with pytest.raises(PoolSaturated):
            svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="quick"),
                      background=False)
        tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="quick"),
                        background=False, admission="side")   # 评测用例不占交互名额
        assert repo.get_task(tid)["status"] == "done"
        assert svc._pending == 1                              # 名额账目不被评测用例改动
    finally:
        svc._release()


def test_compare_runs_while_interactive_gate_full(repo, policy):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    svc = ResearchService(deps, InMemorySaver(), max_pending=1)
    svc._acquire_or_raise()
    try:
        rid = run_compare(svc, [CASE], ["standard", "deep"])  # 名额已满时评测照常完成（含深研专家）
        run = repo.get_eval(rid)
        assert run["status"] == "done", run.get("error")
        assert not [r for r in run["results"] if r.get("error")], run["results"]
        # 评测产生的搜索缓存全部带 eval 前缀 scope，用户侧 RSI 读不到
        keys = [d["key"] for d in repo.db.cache.find({"key": {"$regex": "^ds:search:"}})]
        assert keys and all(_search_key_parts(k)[0].startswith("eval-") for k in keys)
        assert cached_results(repo) == {}
    finally:
        svc._release()


def test_supersede_guard_keeps_user_alerts(repo, policy):
    """评测档案即使被直接传入，也不能按它的时点关闭用户提醒。"""
    svc, services = _alert_services(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="quick"),
              namespace="eval-x", background=False)
    ns_dossier = repo.current_dossier(f"eval-x:{CODE}")
    assert ns_dossier is not None
    services.watch._supersede_open_alerts(ns_dossier)
    assert services.watch.store.get_alert("al_u").status == "open"     # 用户提醒不受评测档案影响
