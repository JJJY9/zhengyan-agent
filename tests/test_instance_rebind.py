"""实例归属与僵尸回收回归：审批恢复重绑实例（回收不再误杀已批准任务）、
收尾 CAS 不复活被回收任务、评测按实例租约回收、停机不停心跳且收尾本实例评测、
对照评测失败不留下「进行中」结论、调度器分钟级回收不依赖扫描节奏。"""
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.evals.compare import run_compare
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.scheduler import Scheduler
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of
from tests.test_phase2 import make_service


def test_resume_rebinds_instance_to_current_process(repo, policy):
    """审批恢复方与创建方不是同一进程：同一 CAS 必须把归属重绑，否则租约回收把已批准任务标失败。"""
    policy.approvals.setdefault("skills", {})["web.search"] = "confirm"
    saver = InMemorySaver()
    svc1 = ResearchService(build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools()), saver)
    tid = svc1.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩与估值", as_of=as_of(), route="standard"),
                     background=False)
    task = repo.get_task(tid)
    assert task["status"] == "waiting_approval"
    repo.db.tasks.update_one({"task_id": tid}, {"$set": {"instance": "inst_ghost"}})   # 创建方进程已退出

    deps2 = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    svc2 = ResearchService(deps2, saver)
    for _ in range(40):
        if repo.get_task(tid)["status"] != "waiting_approval":
            break
        svc2.resume(tid, approve=True, background=False)
    task = repo.get_task(tid)
    assert task["status"] == "done", task
    assert task["instance"] == svc2.instance                     # 归属已重绑到恢复方
    assert repo.recover_stale_tasks(600) == 0                    # 活租约的已批准任务不能被回收
    assert repo.get_task(tid)["status"] == "done"


def test_run_rebinds_ghost_queued_task(repo, policy):
    """排队任务的创建方已退出、由当前实例执行：启动 CAS 同样把归属重绑，之后回收不误杀。"""
    svc = make_service(repo, policy)
    repo.create_task("t_ghost_queued", {"question": "q"}, instance="inst_ghost")
    repo.db.tasks.update_one({"task_id": "t_ghost_queued"}, {"$set": {"status": "queued"}})
    payload = {"task_id": "t_ghost_queued", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "namespace": "",
               "as_of": as_of().isoformat(), "route_requested": "standard", "review_focus": []}
    svc._run("t_ghost_queued", payload)
    task = repo.get_task("t_ghost_queued")
    assert task["status"] == "done", task
    assert task["instance"] == svc.instance


def test_resume_rebind_closes_recovery_window(repo, policy):
    """恢复瞬间就可能撞上另一实例的回收：批准 CAS 本身必须已把归属换成本进程。"""
    svc = make_service(repo, policy)                              # 本进程租约由心跳维持
    repo.create_task("t_win", {"question": "q"}, instance="inst_ghost")
    repo.db.tasks.update_one({"task_id": "t_win"},
                             {"$set": {"status": "waiting_approval", "interrupt": {"kind": "consent"}}})
    assert repo.try_resume_task("t_win", instance=svc.instance)
    assert repo.get_task("t_win")["instance"] == svc.instance     # CAS 内重绑，而不是等 _run
    assert repo.recover_stale_tasks(600) == 0                     # 窗口内回收不能把刚批准的任务标失败
    assert repo.get_task("t_win")["status"] == "running"


def test_finish_task_does_not_resurrect_recovered(repo):
    """收尾写入带 status=running 条件：被回收成 failed 的任务不允许再被改成 done。"""
    repo.create_task("t_recovered", {"question": "q"})
    repo.db.tasks.update_one({"task_id": "t_recovered"}, {"$set": {"status": "failed"}})
    assert repo.finish_task("t_recovered", "done") is False
    assert repo.get_task("t_recovered")["status"] == "failed"


def test_recover_stale_evals_trusts_instance_lease(repo, policy):
    """评测文档带实例归属：属主租约已死即刻回收（不必等 1h），活进程的评测不受影响。"""
    svc = make_service(repo, policy)
    repo.touch_lease(svc.instance)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "rt_ghost", "type": "routing", "status": "running",
                                  "updated_at": now, "instance": "proc_ghost"})
    repo.db.eval_runs.insert_one({"run_id": "rt_live", "type": "routing", "status": "running",
                                  "updated_at": now, "instance": svc.instance})
    assert repo.recover_stale_evals() == 1
    assert repo.get_eval("rt_ghost")["status"] == "failed"
    assert repo.get_eval("rt_live")["status"] == "running"       # 活进程刚登记的评测不能被误杀


def test_shutdown_fails_own_evals_and_keeps_heartbeat(repo, policy):
    """优雅停机：本实例未完成评测标 failed（页面不再永远轮询）；心跳不停，排空中的任务不被新实例当无主。"""
    svc = make_service(repo, policy)
    repo.db.eval_runs.insert_one({"run_id": "rt_own", "type": "routing", "status": "running",
                                  "updated_at": datetime.now(timezone.utc), "instance": svc.instance})
    svc.shutdown()
    assert repo.get_eval("rt_own")["status"] == "failed"
    assert not svc._hb_stop.is_set()


def test_compare_failure_marks_interrupted_verdict(repo, policy):
    """对照评测中途失败：留下的结论必须写明「中断、已完成 n 例」，不能停在「评测进行中」。"""
    svc = make_service(repo, policy)
    real_save = repo.save_eval

    def flaky(doc):
        if doc.get("status") == "running" and doc.get("results"):
            raise RuntimeError("存储瞬时不可用")
        return real_save(doc)

    repo.save_eval = flaky
    try:
        rid = run_compare(svc, [{"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())},
                                {"id": "c2", "question": f"复核 {CODE} 三季度业绩", "ticker": CODE, "as_of": str(as_of())}],
                          ["standard"])
    finally:
        repo.save_eval = real_save
    run = repo.get_eval(rid)
    assert run["status"] == "failed"
    verdict = run["summary"]["verdict"]
    assert verdict["provisional"] is True and "中断" in verdict["text"]


def test_scheduler_periodic_zombie_recovery(monkeypatch):
    """调度器每个快 tick 都回收僵尸任务/评测：不依赖 6 小时扫描节奏。"""
    calls = {"tasks": 0, "evals": 0}

    class R:
        def recover_stale_tasks(self, age, lease_ttl_s=180.0):
            calls["tasks"] += 1

        def recover_stale_evals(self, *a, **k):
            calls["evals"] += 1

    class W:
        def scan_all(self, log=None, is_cancelled=lambda: False):
            return []

    class S:
        def step_all(self, day=None, log=None, is_cancelled=lambda: False):
            return []

    monkeypatch.setattr(Scheduler, "FAST_TICK_S", 0.05)
    sch = Scheduler(SimpleNamespace(deps=SimpleNamespace(repo=R()), watch=W(), strategy=S()), interval_hours=100)
    sch.start()
    try:
        deadline = time.time() + 5
        while calls["tasks"] < 2 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        sch.stop()
    assert calls["tasks"] >= 2 and calls["evals"] >= 2            # 快 tick 独立于扫描间隔
