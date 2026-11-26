"""协作式取消回归：JobBoard 取消标志与结局、RSI 评测/扫描/推盘的取消检查点、取消 API。"""
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from advisor.api.app import create_app
from advisor.api.common import JobBoard
from advisor.harness import TaskCancelled
from advisor.rsi import RSIRunner
from advisor.services import Services
from tests.test_phase2 import make_service
from tests.test_rsi import RSILLM, seed_dossiers


class _ThreadSvc:
    def submit(self, fn):
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        return t


def test_jobboard_cancel_marks_cancelled():
    board = JobBoard()
    seen = {}

    def work(log, is_cancelled):
        for _ in range(500):
            if is_cancelled():
                raise TaskCancelled("作业已取消")
            time.sleep(0.01)
        seen["finished"] = True

    job = board.start(_ThreadSvc(), "test", work)
    time.sleep(0.05)
    assert board.cancel(job["job_id"]) == {"ok": True}
    deadline = time.time() + 5
    while time.time() < deadline and board.get(job["job_id"])["status"] == "running":
        time.sleep(0.02)
    assert board.get(job["job_id"])["status"] == "cancelled"     # 协作点退出而非硬杀
    assert "finished" not in seen


def test_jobboard_cancel_errors():
    board = JobBoard()
    with pytest.raises(HTTPException):
        board.cancel("nope")                                      # 404
    job = board.start(_ThreadSvc(), "test", lambda log, c: None)
    for _ in range(200):
        if board.get(job["job_id"])["status"] != "running":
            break
        time.sleep(0.01)
    with pytest.raises(HTTPException):
        board.cancel(job["job_id"])                               # 已结束：409


def test_evaluate_cancelled_stays_draft(repo, policy):
    seed_dossiers(repo)
    svc = make_service(repo, policy, RSILLM())
    runner = RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)
    runner.collect()
    issue = next(i for i in runner.ledger.list_issues("S1") if i.source["ticker"] == "600001")
    run = runner.propose("S1", issue.id)
    with pytest.raises(TaskCancelled):
        runner.evaluate(run.run_id, is_cancelled=lambda: True)
    assert runner.ledger.get_run(run.run_id).status == "draft"    # 取消不写半成品，可重新评测


def test_scan_all_cancelled_between_subs(repo, policy):
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    services.watch.store.subscribe("600001", "甲", 10)
    services.watch.store.subscribe("000002", "乙", 10)
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(TaskCancelled):
        services.watch.scan_all(log=lambda m: None, is_cancelled=cancelled)


def test_step_all_cancelled_before_work(repo, policy):
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    with pytest.raises(TaskCancelled):
        services.strategy.step_all(is_cancelled=lambda: True)


def test_cancel_endpoint(repo, policy):
    svc = make_service(repo, policy)
    with TestClient(create_app(deps=svc.deps, checkpointer=None)) as c:
        board = c.app.state.jobs
        board.jobs["job_x"] = {"job_id": "job_x", "kind": "t", "key": None, "status": "running", "log": [],
                               "result": None, "error": None, "cancel": False, "started_at": None}
        assert c.post("/api/jobs/job_x/cancel").status_code == 200
        assert c.post("/api/jobs/job_x/cancel").status_code == 409     # 已请求过取消
        assert c.post("/api/jobs/nope/cancel").status_code == 404
        board.jobs["job_y"] = dict(board.jobs["job_x"], job_id="job_y", status="done", log=[], cancel=False)
        assert c.post("/api/jobs/job_y/cancel").status_code == 409     # 已结束
