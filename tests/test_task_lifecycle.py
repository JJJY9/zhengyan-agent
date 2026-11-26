"""任务生命周期回归：租约回收不误杀排队任务、CAS 启动不复活、原子取消不回写旧状态、
复核原子领取与 sync 容错占位、停机取消排队任务、调度器协作停机。"""
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from advisor.harness import TaskCancelled
from advisor.scheduler import Scheduler
from advisor.services import Services
from advisor.watch.models import Alert
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service


def _mk_task(repo, tid, status="queued", age_s=0, instance=None):
    repo.create_task(tid, {"question": "q"}, instance=instance)
    repo.db.tasks.update_one({"task_id": tid}, {"$set": {
        "status": status, "updated_at": datetime.now(timezone.utc) - timedelta(seconds=age_s)}})


def test_recovery_trusts_lease_not_age(repo, policy):
    svc = make_service(repo, policy)
    repo.touch_lease(svc.instance)
    _mk_task(repo, "t_live_queued", status="queued", age_s=7200, instance=svc.instance)
    _mk_task(repo, "t_dead_fresh", status="running", age_s=5, instance="inst_ghost")
    _mk_task(repo, "t_legacy_old", status="queued", age_s=7200, instance=None)
    _mk_task(repo, "t_legacy_fresh", status="queued", age_s=5, instance=None)
    assert repo.recover_stale_tasks(600) == 2
    assert repo.get_task("t_live_queued")["status"] == "queued"    # 活进程的排队任务（两小时没动静）不能误杀
    assert repo.get_task("t_dead_fresh")["status"] == "failed"     # 无租约进程的任务刚更新也算无主
    assert repo.get_task("t_legacy_old")["status"] == "failed"
    assert repo.get_task("t_legacy_fresh")["status"] == "queued"


def test_run_refuses_recovered_task(repo, policy):
    svc = make_service(repo, policy)
    _mk_task(repo, "t_failed", status="failed", instance=svc.instance)
    payload = {"task_id": "t_failed", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "namespace": "",
               "as_of": as_of().isoformat(), "route_requested": "standard", "review_focus": []}
    svc._run("t_failed", payload)
    assert repo.get_task("t_failed")["status"] == "failed"        # 已被回收的任务绝不复活执行


def test_cancel_does_not_resurrect_finished_task(repo):
    repo.create_task("t_done", {})
    repo.update_task("t_done", status="done")
    repo.get_task = lambda task_id: {"status": "running"}          # 模拟读完状态后任务才完成
    assert repo.request_cancel("t_done") is None                  # 原子条件更新：不再回写旧状态
    assert repo.db.tasks.find_one({"task_id": "t_done"})["status"] == "done"


def test_review_claim_atomic_and_sync_forgiving(repo, policy):
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    store = services.watch.store
    store.subscribe(CODE, "示例酒业", 10)
    store.add_alert(Alert(id="al1", ticker=CODE, name="示例酒业", dossier_key=CODE, dossier_version=1,
                          fingerprint="fp1", events=[], affected=[], created_at=datetime.now(timezone.utc)))
    assert store.claim_review("al1")
    assert store.get_alert("al1").review_task_id == "pending"
    services.watch.sync()                                          # 领取后正在创建任务：占位新鲜，sync 不能重置
    assert store.get_alert("al1").status == "reviewing"
    assert not store.claim_review("al1")                           # 双击领取失败
    store.finish_review("al1", "t_ok")
    assert store.get_alert("al1").review_task_id == "t_ok"
    store.update_alert("al1", status="reviewing", review_task_id="pending",
                       review_started_at=(datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat())
    services.watch.sync()                                          # 占位过期从未回填：放回待处理
    assert store.get_alert("al1").status == "open"


def test_shutdown_cancels_queued_tasks(repo, policy):
    svc = make_service(repo, policy)
    _mk_task(repo, "t_q", status="queued", instance=svc.instance)
    svc.shutdown()
    assert repo.get_task("t_q")["status"] == "cancelled"           # 停机后注定不会执行的任务不再悬挂


def test_scheduler_stop_waits_and_cancels_tick():
    started = threading.Event()
    state = {"cancelled": False, "steps": 0}

    class W:
        def scan_all(self, log=None, is_cancelled=lambda: False):
            started.set()
            for _ in range(200):
                if is_cancelled():
                    state["cancelled"] = True
                    raise TaskCancelled("停机")
                time.sleep(0.02)
            return []

    class S:
        def step_all(self, day=None, log=None, is_cancelled=lambda: False):
            state["steps"] += 1
            return []

    sch = Scheduler(SimpleNamespace(watch=W(), strategy=S()), interval_hours=0.0001)
    sch.start()
    assert started.wait(timeout=5)
    t0 = time.time()
    sch.stop()
    assert not sch._thread.is_alive()                              # stop 在检查点 join，而不是直接放掉线程
    assert time.time() - t0 < 5
    assert state["cancelled"] and state["steps"] == 0              # 扫描在检查点取消，推盘未开始
