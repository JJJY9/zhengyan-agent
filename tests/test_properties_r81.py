"""R81a 性质测试：订阅提醒状态机的随机操作序列不变量。

合法状态空间：open（待处理）/ reviewing（复核中，必有 task 或 pending 占位）/
resolved（复核完成，review_task_id 为 done 任务且 resolved_version 与任务结果一致）/ dismissed。
随机交替 claim/finish/release/sync/dismiss 与任务结局（done/failed），断言：
1) 每步后状态属于枚举，且 reviewing ⇒ 有 task 占位；
2) resolved ⇒ 绑定任务是 done 且 resolved_version 等于任务结果版本；
3) open/dismissed ⇒ 不残留任务占位；
4) 非法操作（非 open 领取/忽略）抛 WatchError 且状态不变；
5) 活实例的 pending 占位再久也不被 sync 重置；死实例且过宽限才放回 open。"""
import random
from datetime import datetime, timedelta, timezone

import pytest

from advisor.services import Services
from advisor.watch.models import Alert
from advisor.watch.scanner import WatchError
from tests.fakes import CODE
from tests.test_phase2 import make_service


def _mk_task(repo, tid, status, version=1):
    repo.create_task(tid, {"question": "q"})
    repo.db.tasks.update_one({"task_id": tid}, {"$set": {
        "status": status, "instance": None,
        "result": {"version": version} if status == "done" else None}})


def _alerts(repo, policy, n=3):
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    store = services.watch.store
    store.subscribe(CODE, "示例酒业", 10)
    for i in range(n):
        store.add_alert(Alert(id=f"al{i}", ticker=CODE, name="示例酒业", dossier_key=CODE, dossier_version=1,
                              fingerprint=f"fp{i}", events=[], affected=[],
                              created_at=datetime.now(timezone.utc)))
    return svc, services, [f"al{i}" for i in range(n)]


def _check(store, repo, aid):
    a = store.get_alert(aid)
    assert a.status in ("open", "reviewing", "resolved", "dismissed")
    if a.status == "reviewing":
        assert a.review_task_id, a
    if a.status == "resolved":
        task = repo.get_task(a.review_task_id)
        assert task and task["status"] == "done", (aid, task)
        assert a.resolved_version == task["result"]["version"]
    if a.status in ("open", "dismissed"):
        assert a.review_task_id is None


def test_watch_alert_state_machine_properties(repo, policy):
    rng = random.Random(20261201)
    svc, services, ids = _alerts(repo, policy)
    store, watch = services.watch.store, services.watch
    _mk_task(repo, "t_done", "done", version=7)
    _mk_task(repo, "t_fail", "failed")
    tasks = ["t_done", "t_fail"]
    tid_seq = 0

    for _ in range(60):
        aid = rng.choice(ids)
        before = store.get_alert(aid)
        op = rng.choice(["claim", "finish", "release", "sync", "dismiss", "advance"])
        if op == "claim":
            ok = store.claim_review(aid, instance=svc.instance)
            assert ok == (before.status == "open")                 # 只有 open 能被领取
            if not ok:
                assert store.get_alert(aid).status == before.status
        elif op == "finish":
            tid_seq += 1
            tid = rng.choice(tasks) if rng.random() < 0.5 else f"t_x{aid}_{tid_seq}"
            if tid not in ("t_done", "t_fail") and not repo.get_task(tid):
                _mk_task(repo, tid, rng.choice(["done", "failed"]), version=tid_seq)
            store.finish_review(aid, tid)
        elif op == "release":
            store.release_review(aid)
        elif op == "dismiss":
            if before.status == "open":
                watch.dismiss(aid, "性质测试忽略原因")
            else:
                with pytest.raises(WatchError):
                    watch.dismiss(aid, "性质测试忽略原因")
                assert store.get_alert(aid).status == before.status
        elif op == "sync":
            watch.sync()
        else:  # advance：随机推进复核任务的结局，再交给 sync 收敛
            a = store.get_alert(aid)
            if a.status == "reviewing" and a.review_task_id and not a.review_task_id.startswith("t_done") \
                    and not a.review_task_id.startswith("t_fail"):
                repo.db.tasks.update_one({"task_id": a.review_task_id},
                                         {"$set": {"status": rng.choice(["done", "failed"]),
                                                   "result": {"version": 3}}})
        _check(store, repo, aid)


def test_live_placeholder_survives_dead_placeholder_resets(repo, policy):
    """活租约的 pending 占位再久不被重置；死租约且过宽限才放回 open（存量语义的性质化断言）。"""
    svc, services, ids = _alerts(repo, policy, n=2)
    store = services.watch.store
    live, dead = ids
    svc.deps.repo.touch_lease(svc.instance)
    assert store.claim_review(live, instance=svc.instance)
    assert store.claim_review(dead, instance="inst_ghost")
    old = (datetime.now(timezone.utc) - timedelta(seconds=900)).isoformat()
    store.update_alert(live, review_started_at=old)
    store.update_alert(dead, review_started_at=old)
    services.watch.sync()
    assert store.get_alert(live).status == "reviewing"
    assert store.get_alert(dead).status == "open"
    _check(store, svc.deps.repo, dead)
