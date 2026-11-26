"""R59a 修复窗口自审回归：租约误判时活执行者成果修复、回收方自排除、
评测归属与写回防护、失败评测结论改写、证据冲突白名单、并发注记基线提前。"""

from fastapi.testclient import TestClient

from advisor.api.app import create_app
from advisor.contracts import ResearchRequest
from advisor.evals.compare import run_compare
from advisor.services import Services
from tests.fakes import CODE, as_of
from tests.test_diff_evals_hardening import _card
from tests.test_phase2 import make_service


def _running_task(repo, tid, instance):
    repo.create_task(tid, {"question": "q"}, instance=instance)
    repo.db.tasks.update_one({"task_id": tid}, {"$set": {"status": "running"}})


def test_finish_repairs_misjudged_recovery(repo, policy):
    """心跳断更（休眠/时钟跳变/Mongo 短断）让租约过期：回收可能误判，但执行者仍带着完整结果。"""
    svc = make_service(repo, policy)
    _running_task(repo, "t_misjudged", svc.instance)
    repo.db.leases.delete_many({})                                # 租约过期
    assert repo.recover_stale_tasks(600) == 1
    assert repo.get_task("t_misjudged")["status"] == "failed"
    ok = repo.finish_task("t_misjudged", "done", instance=svc.instance, repair_recovered=True,
                          result={"version": 3})
    assert ok is True
    task = repo.get_task("t_misjudged")
    assert task["status"] == "done" and task["result"]["version"] == 3   # 活执行者的成果不能被丢弃


def test_finish_repair_requires_ownership(repo, policy):
    """修复通道只对任务归属方开放：他实例不能借误判恢复写终态。"""
    svc = make_service(repo, policy)
    _running_task(repo, "t_other", "inst_other")
    repo.db.leases.delete_many({})
    assert repo.recover_stale_tasks(600) == 1
    assert repo.finish_task("t_other", "done", instance=svc.instance, repair_recovered=True) is False
    assert repo.get_task("t_other")["status"] == "failed"


def test_recovery_excludes_own_instance(repo, policy):
    """回收方绝不回收自己：快 tick 与心跳竞争时不能把自己正在执行的任务判死。"""
    svc = make_service(repo, policy)
    _running_task(repo, "t_self", svc.instance)
    repo.db.leases.delete_many({})
    touched: list[str] = []
    real_touch = repo.touch_lease
    repo.touch_lease = lambda inst: touched.append(inst)          # 记录但不写租约：单独验证「跳过自己」
    try:
        assert repo.recover_stale_tasks(600, exclude_instance=svc.instance) == 0
    finally:
        repo.touch_lease = real_touch
    assert repo.get_task("t_self")["status"] == "running"
    assert svc.instance in touched                                # 回收前先续自己的租约


def test_recovery_rewrites_interrupted_verdict(repo, policy):
    """回收/停机标 failed 的评测必须同时改写临时结论：失败行不能再显示「评测进行中」。"""
    svc = make_service(repo, policy)
    verdict = {"decision": "insufficient", "provisional": True, "text": "评测进行中：结论在全部用例完成后给出"}
    repo.db.eval_runs.insert_one({"run_id": "rt_ghost2", "type": "compare", "status": "running",
                                  "instance": "proc_ghost", "results": [{"case": "a"}, {"case": "b"}],
                                  "summary": {"verdict": verdict}})
    repo.db.eval_runs.insert_one({"run_id": "rt_mine", "type": "compare", "status": "running",
                                  "instance": svc.instance, "results": [{"case": "a"}],
                                  "summary": {"verdict": verdict}})
    assert repo.recover_stale_evals() == 1
    doc = repo.get_eval("rt_ghost2")
    assert doc["status"] == "failed" and "已完成 2 例" in doc["summary"]["verdict"]["text"]
    assert repo.fail_instance_evals(svc.instance, "服务停止，评测中断；可重新发起") == 1
    mine = repo.get_eval("rt_mine")
    assert "中断" in mine["summary"]["verdict"]["text"]


def test_watch_sync_excludes_own_instance(repo, policy):
    """扫描同步内置的僵尸回收同样排除本进程实例。"""
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    seen = {}
    real = repo.recover_stale_tasks

    def spy(max_age_s, lease_ttl_s=180.0, exclude_instance=None):
        seen["exclude"] = exclude_instance
        return real(max_age_s, lease_ttl_s=lease_ttl_s, exclude_instance=exclude_instance)

    repo.recover_stale_tasks = spy
    try:
        services.watch.sync()
    finally:
        repo.recover_stale_tasks = real
    assert seen["exclude"] == svc.deps.instance


def test_eval_api_precreate_carries_instance(repo, policy):
    """API 预创建的评测文档带归属：排队中就被停机取消的评测也能被回收命中。"""
    svc = make_service(repo, policy)
    saves: list[dict] = []
    real = repo.save_eval

    def spy(doc, **kw):
        saves.append(dict(doc))
        return real(doc, **kw)

    repo.save_eval = spy
    try:
        with TestClient(create_app(deps=svc.deps, checkpointer=None)) as c:
            c.post("/api/evals/routing")
    finally:
        repo.save_eval = real
    assert saves and saves[0].get("instance") == svc.deps.instance   # 首次写入（API 预创建）就带归属


def test_compare_does_not_write_done_after_shutdown(repo, policy):
    """停机把评测标 failed 后，作业的下一次写回不能把它复活成 running/done。"""
    svc = make_service(repo, policy)
    real = repo.save_eval
    fired = {"n": 0}

    def hook(doc, **kw):
        out = real(doc, **kw)
        if doc.get("status") == "running" and doc.get("results") and fired["n"] == 0:
            fired["n"] = 1
            repo.fail_instance_evals(svc.instance, "服务停止，评测中断；可重新发起")
        return out

    repo.save_eval = hook
    try:
        rid = run_compare(svc, [{"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())},
                                {"id": "c2", "question": f"复核 {CODE} 三季度业绩", "ticker": CODE, "as_of": str(as_of())}],
                          ["standard"])
    finally:
        repo.save_eval = real
    run = repo.get_eval(rid)
    assert run["status"] == "failed"                              # 不能覆盖停机终态
    assert "中断" in run["summary"]["verdict"]["text"]


def test_put_evidence_ignores_session_metadata(repo):
    """检索会话元数据（摘要/重排分/注意事项/簇号/等级）随查询变化，不是证据冲突。"""
    a = _card("body-session", title="同一标题")
    assert repo.put_evidence(a) is True
    b = a.model_copy(update={"excerpt": "不同摘要", "rerank_score": 9.9, "caveats": ["x"],
                             "cluster_id": "c9", "level": "C"})
    assert repo.put_evidence(b) is True
    assert repo.db.evidence.find_one({"id": a.id})["excerpt"] == a.excerpt   # 旧卡仍生效


def test_commit_notes_parent_drift_before_first_insert(repo, policy):
    """路由后、首次提交前他人已提交：即便首次插入成功，档案也要自述对照版本漂移。"""
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    v1 = repo.get_dossier(CODE)
    assert v1.version == 1
    real = repo.latest_dossier_version_at
    calls = {"n": 0}

    def drifting(key, day):
        calls["n"] += 1
        return 2 if calls["n"] > 1 else real(key, day)   # collect 载入 prior=v1，提交时看到 v2

    repo.latest_dossier_version_at = drifting
    try:
        svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    finally:
        repo.latest_dossier_version_at = real
    latest = repo.get_dossier(CODE)
    assert any("由 v1 变为 v2" in n for n in latest.boundary_notes)
