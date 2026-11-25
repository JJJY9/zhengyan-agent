"""订阅扫描与崩溃恢复回归：当前档案按 as_of 取（回填不高抬版本）、僵尸任务/提醒自动恢复。"""
from datetime import date, datetime, timedelta, timezone

from advisor.contracts import ResearchRequest
from advisor.services import Services
from advisor.watch.models import Alert
from tests.fakes import CODE
from tests.test_phase2 import make_service


def _research(svc, as_of: date) -> int:
    tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of, route="standard"), background=False)
    task = svc.deps.repo.get_task(tid)
    assert task["status"] == "done", task
    return task["result"]["version"]


def _alert(aid: str, version: int, fingerprint: str) -> Alert:
    return Alert(id=aid, ticker=CODE, name="示例酒业", dossier_key=CODE, dossier_version=version,
                 fingerprint=fingerprint, events=[], created_at=datetime.now(timezone.utc))


# ---------- 当前档案按研究时点取 ----------

def test_scan_uses_latest_as_of_dossier_not_highest_version(repo, policy):
    """CLI --as-of 回填产出「版本更高、as_of 更早」的档案：扫描不能把回填当最新，否则误报新财报。"""
    svc = make_service(repo, policy)
    v1 = _research(svc, date(2026, 10, 18))
    v2 = _research(svc, date(2026, 6, 1))
    assert v2 > v1, (v1, v2)                       # 回填的版本号更高、研究时点更早
    services = Services(svc.deps, svc)
    assert repo.current_dossier(CODE).version == v1
    assert repo.current_dossier(CODE, as_of=date(2026, 6, 1)).version == v2
    alert = services.watch.scan(CODE, today=date(2026, 10, 18))
    assert alert is None or (alert.dossier_version == v1 and all(e.kind != "new_report" for e in alert.events))


def test_supersede_compares_by_as_of_not_version(repo, policy):
    """挂在回填版本上的旧提醒：已有更新时点的档案后必须自动关闭，而不是按版本号比较被保留。"""
    svc = make_service(repo, policy)
    _research(svc, date(2026, 10, 18))            # 先有 as_of 最新的档案
    v2 = _research(svc, date(2026, 6, 1))         # 再回填出高版本旧时点档案
    services = Services(svc.deps, svc)
    services.watch.store.add_alert(_alert("al_backfill", v2, "fp-backfill"))
    services.watch.scan(CODE, today=date(2026, 10, 18))
    assert services.watch.store.get_alert("al_backfill").status == "dismissed"


# ---------- 崩溃后僵尸任务与提醒恢复 ----------

def test_stale_review_task_recovers_and_unlocks_alert(repo, policy):
    service = make_service(repo, policy)
    services = Services(service.deps, service)
    now = datetime.now(timezone.utc)
    repo.create_task("t_zombie", {"question": "复核 600999"})
    repo.db.tasks.update_one({"task_id": "t_zombie"},
                             {"$set": {"status": "running", "updated_at": now - timedelta(seconds=3600)}})
    repo.create_task("t_live", {"question": "复核 600999"})          # 新鲜任务不能被误杀
    repo.db.tasks.update_one({"task_id": "t_live"}, {"$set": {"status": "running"}})
    services.watch.store.add_alert(_alert("al_rev", 1, "fp-review"))
    services.watch.store.update_alert("al_rev", status="reviewing", review_task_id="t_zombie")
    services.watch.sync()
    assert repo.get_task("t_zombie")["status"] == "failed"
    assert repo.get_task("t_live")["status"] == "running"
    a = services.watch.store.get_alert("al_rev")
    assert a.status == "open" and a.review_task_id is None      # 提醒解锁，可重新发起复核或忽略


def test_startup_recovery_marks_legacy_running_tasks(repo, policy):
    service = make_service(repo, policy)
    repo.create_task("t_left", {"question": "复核 600999"})
    repo.db.tasks.update_one({"task_id": "t_left"},
                             {"$set": {"status": "running",
                                       "updated_at": datetime.now(timezone.utc) - timedelta(seconds=3600)}})
    Services(service.deps, service)                               # 模拟进程重启
    assert repo.get_task("t_left")["status"] == "failed"
    assert "重新发起" in (repo.get_task("t_left").get("error") or "")
