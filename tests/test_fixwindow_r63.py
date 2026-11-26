"""R63a 收敛性三审回归：排队评测不做年龄回收（准入认领后才有停滞兜底）、
失败文档不覆盖并发写入的 routes、认领写 started_at、并发注记不再出现 vNone。"""
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from advisor.api.app import create_app
from advisor.contracts import ResearchRequest
from advisor.evals.compare import run_compare
from advisor.evals.routing import run_routing_eval
from advisor.graph.runner import ResearchService
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service

CASE = {"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())}


def test_queued_eval_not_killed_by_age(repo, policy):
    """排队未开跑的评测（started_at=None）只信租约：按年龄回收会误杀仍在排队的活作业。"""
    svc = make_service(repo, policy)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "ev_q", "type": "compare", "status": "running",
                                  "instance": svc.instance, "started_at": None,
                                  "updated_at": now - timedelta(hours=3), "results": [],
                                  "routes": ["standard"], "cases": [CASE]})
    assert repo.recover_stale_evals() == 0
    assert repo.get_eval("ev_q")["status"] == "running"


def test_worker_claim_marks_started(repo, policy):
    """出队认领写 started_at：开跑后的停滞才进入兜底回收口径。"""
    svc = make_service(repo, policy)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "ev_q2", "type": "compare", "status": "running",
                                  "instance": svc.instance, "started_at": None,
                                  "updated_at": now - timedelta(hours=3), "results": [],
                                  "routes": ["standard"], "cases": [CASE]})
    called = {"n": 0}
    real = svc.start

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    svc.start = spy
    try:
        run_compare(svc, [CASE], ["standard"], run_id="ev_q2")
    finally:
        svc.start = real
    doc = repo.get_eval("ev_q2")
    assert doc["started_at"] is not None and called["n"] == 1
    assert doc["status"] == "done"


def test_routing_claim_marks_started(repo, policy):
    svc = make_service(repo, policy)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "rt_q", "type": "routing", "status": "running",
                                  "instance": svc.deps.instance, "started_at": None,
                                  "updated_at": now - timedelta(hours=3), "results": []})
    run_routing_eval(svc.deps, run_id="rt_q")
    assert repo.get_eval("rt_q")["started_at"] is not None


def test_eval_api_precreate_marks_queued(repo, policy, monkeypatch):
    """API 预创建显式写 started_at=None（排队态），停滞兜底不会把排队中的评测标失败。"""
    svc = make_service(repo, policy)
    monkeypatch.setattr(ResearchService, "submit_side", lambda self, fn, *a, **k: "stub")
    with TestClient(create_app(deps=svc.deps, checkpointer=None)) as c:
        rid = c.post("/api/evals/routing").json()["run_id"]
    doc = repo.get_eval(rid)
    assert "started_at" in doc and doc["started_at"] is None
    repo.db.eval_runs.update_one({"run_id": rid}, {"$set": {"updated_at": datetime.now(timezone.utc) - timedelta(hours=3)}})
    assert repo.recover_stale_evals() == 0


def test_started_stalled_recovered_with_reason(repo, policy):
    """已开跑且停滞的评测按停滞兜底回收，error 归因「停滞」而非「进程已不在」。"""
    svc = make_service(repo, policy)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "rt_s", "type": "routing", "status": "running",
                                  "instance": svc.instance, "started_at": now - timedelta(hours=3),
                                  "updated_at": now - timedelta(hours=3), "results": []})
    assert repo.recover_stale_evals() == 1
    doc = repo.get_eval("rt_s")
    assert doc["status"] == "failed" and "停滞" in doc["error"]


def test_fail_docs_does_not_regress_routes(repo, policy):
    """worker 进度写回与回收交错：routes 不能被调用方旧快照整段覆写打回。"""
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "ev_race", "type": "compare", "status": "running",
                                  "instance": "proc_ghost", "results": [{"case": "a"}],
                                  "updated_at": now, "routes": ["standard"],
                                  "summary": {"routes": {"standard": {"cases": 1}},
                                              "verdict": {"decision": "insufficient", "provisional": True,
                                                          "text": "评测进行中：结论在全部用例完成后给出"}}})
    real = repo.db.eval_runs.update_one
    fired = {"n": 0}

    def hook(filt, upd, *a, **k):
        if isinstance(upd, dict) and "summary.verdict" in (upd.get("$set") or {}) and fired["n"] == 0:
            fired["n"] = 1
            real({"run_id": "ev_race"}, {"$set": {
                "results": [{"case": "a"}, {"case": "b"}, {"case": "c"}],
                "summary": {"routes": {"standard": {"cases": 3}},
                            "verdict": {"decision": "insufficient", "provisional": True,
                                        "text": "评测进行中：结论在全部用例完成后给出"}}}})
        return real(filt, upd, *a, **k)

    repo.db.eval_runs.update_one = hook
    try:
        assert repo.recover_stale_evals() == 1
    finally:
        repo.db.eval_runs.update_one = real
    doc = repo.get_eval("ev_race")
    assert doc["status"] == "failed"
    assert doc["summary"]["routes"]["standard"]["cases"] == 3      # 并发写入的 routes 不被打回
    assert "中断" in doc["summary"]["verdict"]["text"]


def test_retry_note_avoids_vnone(repo, policy):
    """首版并发出现在重试循环内：注记不能写成「由 vNone 变为 v1」。"""
    svc = make_service(repo, policy)
    real_try = repo.try_insert_dossier
    fired = {"n": 0}

    def racing(dossier):
        if fired["n"] == 0:
            fired["n"] = 1
            repo.insert_dossier(dossier.model_copy(update={"version": 1, "parent_version": None,
                                                           "task_id": "t_rival", "report_md": "竞争首版"}))
            return False
        return real_try(dossier)

    repo.try_insert_dossier = racing
    try:
        svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                  background=False)
    finally:
        repo.try_insert_dossier = real_try
    latest = repo.get_dossier(CODE)
    assert latest.version == 2 and latest.parent_version == 1
    notes = latest.boundary_notes
    assert notes and all("vNone" not in n for n in notes)
    assert any("未做上一版本对照" in n and "v1" in n for n in notes)
    assert "vNone" not in latest.report_md
