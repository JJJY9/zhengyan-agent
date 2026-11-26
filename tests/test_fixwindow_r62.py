"""R62a 修复窗口二审回归：失败评测 summary 完整形状、首笔条件认领（不复活终态评测）、
修复通道清理旧 error、评测停滞兜底、gather 恒引用库卡、prior 未载入的并发注记。"""
from datetime import datetime, timedelta, timezone

from advisor.contracts import ResearchRequest
from advisor.evals.compare import run_compare
from advisor.evals.routing import run_routing_eval
from advisor.graph.deps import build_deps
from advisor.graph.gather import gather_web
from advisor.harness import Budget
from tests.fakes import CODE, FakeLLM, FakeSource, as_of
from tests.test_phase2 import make_service

CASE = {"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())}


def _running_task(repo, tid, instance):
    repo.create_task(tid, {"question": "q"}, instance=instance)
    repo.db.tasks.update_one({"task_id": tid}, {"$set": {"status": "running"}})


def test_failed_eval_summary_has_full_shape(repo, policy):
    """失败评测的 summary 必须带 routes 形状：点路径只写 verdict 会让详情页按 s.routes 崩溃。"""
    svc = make_service(repo, policy)
    repo.db.eval_runs.insert_one({"run_id": "ev_bare", "type": "compare", "status": "running",
                                  "instance": svc.instance, "results": [{"case": "a"}]})
    assert repo.fail_instance_evals(svc.instance, "服务停止，评测中断；可重新发起") == 1
    doc = repo.get_eval("ev_bare")
    assert doc["status"] == "failed"
    assert doc["summary"]["routes"] == {} and "已完成 1 例" in doc["summary"]["verdict"]["text"]


def test_compare_skips_precreated_failed_eval(repo, policy):
    """已判失败的评测作业不能再开跑：首笔写回先做条件认领，终态文档不复活。"""
    svc = make_service(repo, policy)
    repo.db.eval_runs.insert_one({"run_id": "ev_dead", "type": "compare", "status": "failed",
                                  "instance": svc.instance, "results": [], "error": "服务停止"})
    called = {"n": 0}
    real_start = svc.start

    def spy(*a, **k):
        called["n"] += 1
        return real_start(*a, **k)

    svc.start = spy
    try:
        rid = run_compare(svc, [CASE], ["standard"], run_id="ev_dead")
    finally:
        svc.start = real_start
    assert rid == "ev_dead" and called["n"] == 0
    doc = repo.get_eval("ev_dead")
    assert doc["status"] == "failed" and doc["error"] == "服务停止"


def test_routing_skips_precreated_failed_eval(repo, policy, tmp_path):
    """路由评测同样先认领：已失败文档不被首笔无条件写复活，error 不被覆盖。"""
    svc = make_service(repo, policy)
    repo.db.eval_runs.insert_one({"run_id": "rt_dead", "type": "routing", "status": "failed",
                                  "instance": svc.deps.instance, "error": "服务停止，评测中断"})
    rid = run_routing_eval(svc.deps, path=tmp_path / "missing.json", run_id="rt_dead")
    assert rid == "rt_dead"
    doc = repo.get_eval("rt_dead")
    assert doc["status"] == "failed" and doc["error"] == "服务停止，评测中断"


def test_repair_clears_stale_error_on_waiting(repo, policy):
    """租约误判修复回待审批：旧「已标记失败」的 error 必须清掉，审批卡不能与红字同屏。"""
    svc = make_service(repo, policy)
    _running_task(repo, "t_rep", svc.instance)
    repo.db.leases.delete_many({})
    assert repo.recover_stale_tasks(600) == 1
    assert repo.get_task("t_rep")["error"]
    assert repo.finish_task("t_rep", "waiting_approval", instance=svc.instance, repair_recovered=True,
                            interrupt={"kind": "consent"})
    task = repo.get_task("t_rep")
    assert task["status"] == "waiting_approval" and task["error"] is None


def test_resume_clears_stale_error(repo, policy):
    """审批恢复转 running 时清历史 error：运行中的任务不携带「已被回收」文案。"""
    svc = make_service(repo, policy)
    repo.create_task("t_resume_err", {"question": "q"}, instance=svc.instance)
    repo.db.tasks.update_one({"task_id": "t_resume_err"},
                             {"$set": {"status": "waiting_approval", "error": "任务所属进程已不在（崩溃或无租约），已标记失败；可重新发起",
                                       "interrupt": {"kind": "consent"}}})
    assert repo.try_resume_task("t_resume_err", instance=svc.instance)
    assert repo.get_task("t_resume_err")["error"] is None


def test_recover_stale_evals_stalled_under_alive_instance(repo, policy):
    """活进程名下的死作业（首笔 DB 调用即异常）也要按停滞兜底回收，不能永久 running。"""
    svc = make_service(repo, policy)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "rt_stall", "type": "routing", "status": "running",
                                  "instance": svc.instance, "updated_at": now - timedelta(hours=3),
                                  "results": []})
    repo.db.eval_runs.insert_one({"run_id": "rt_fresh", "type": "routing", "status": "running",
                                  "instance": svc.instance, "updated_at": now, "results": []})
    assert repo.recover_stale_evals() == 1
    assert repo.get_eval("rt_stall")["status"] == "failed"
    assert repo.get_eval("rt_fresh")["status"] == "running"


class _SummaryWeb:
    """同一正文/标题，摘要（excerpt）随查询变化：put_evidence 幂等返回 True 但新元数据不落库。"""

    def __init__(self):
        self.summary = "来源：证券时报 示例酒业上半年实现营业收入910亿元。"

    def search(self, query, count):
        return [{"title": "旧标题", "url": "https://www.stcn.com/a/1.html", "site_name": "证券时报",
                 "authority": 5, "rerank_score": 2.5, "date": "2026-08-16T08:00:00+00:00",
                 "summary": self.summary}]

    def fetch(self, url):
        return {"title": "正文", "content": "<p>示例酒业上半年实现营业收入910亿元。</p>"}


def test_gather_references_stored_card_even_when_put_true(repo, policy):
    """put_evidence 返回 True（会话元数据不同、新卡未落库）时，报告也必须引用库内生效卡。"""
    web = _SummaryWeb()
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=web)
    h1 = deps.harness("t_s1", "standard", as_of(), Budget(limits=policy.route("standard").budget))
    r1 = gather_web(h1, deps, ["示例酒业 业绩"], as_of(), "standard", max_fetch=2)
    stored_excerpt = r1.cards[0].excerpt
    web.summary = "来源：证券时报 示例酒业上半年实现营业收入910亿元，同比增长1.11%。"
    h2 = deps.harness("t_s2", "standard", as_of(), Budget(limits=policy.route("standard").budget))
    r2 = gather_web(h2, deps, ["示例酒业 最新业绩"], as_of(), "standard", max_fetch=2)   # 换查询绕开技能缓存
    assert r2.cards[0].excerpt == stored_excerpt          # 引用与库一致，而非未落库的新摘要
    assert not any("不一致" in u for u in r2.unknowns)     # 会话元数据差异不报冲突


def test_commit_notes_concurrent_first_version(repo, policy):
    """研究开始时无上一版、提交时他人已提交：档案自述未做上一版本对照。"""
    svc = make_service(repo, policy)
    real = repo.latest_dossier_version_at
    calls = {"n": 0}

    def drifting(key, day):
        calls["n"] += 1
        return 1 if calls["n"] > 1 else None      # collect 无上一版；提交时已有 v1

    repo.latest_dossier_version_at = drifting
    try:
        svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                  background=False)
    finally:
        repo.latest_dossier_version_at = real
    d = repo.get_dossier(CODE)
    assert any("研究开始时无上一版本" in n and "v1" in n for n in d.boundary_notes)
