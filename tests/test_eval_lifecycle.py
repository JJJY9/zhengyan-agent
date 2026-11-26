"""评测生命周期回归：运行中只写临时结论、金标读取失败落 failed、启动回收评测僵尸、
RSI 继承统计排除评测档案。"""
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from advisor.api.app import create_app
from advisor.evals.compare import run_compare
from advisor.evals.routing import run_routing_eval
from advisor.services import Services
from tests.fakes import CODE
from tests.test_phase2 import make_service

CASE = {"id": "x", "question": f"复核 {CODE}", "ticker": CODE, "as_of": "2026-10-18"}


def test_compare_running_doc_has_provisional_verdict_only(repo, policy):
    svc = make_service(repo, policy)
    seen: list[dict] = []
    real = repo.save_eval

    def spy(run, **kw):
        seen.append({**run, "results": list(run.get("results") or [])})
        return real(run, **kw)

    repo.save_eval = spy
    try:
        run_compare(svc, [CASE], ["standard"])
    finally:
        repo.save_eval = real
    running = [d for d in seen if d.get("status") == "running" and d.get("summary")]
    assert running, "进度应逐例落库"
    for d in running:
        v = d["summary"]["verdict"]
        assert v.get("provisional") and v["decision"] == "insufficient"   # 运行中不给 adopt/keep 定稿
    final = seen[-1]
    assert final["status"] == "done" and not final["summary"]["verdict"].get("provisional")


def test_routing_gold_read_failure_lands_failed(repo, policy, tmp_path):
    svc = make_service(repo, policy)
    try:
        run_routing_eval(svc.deps, path=tmp_path / "missing.json", run_id="rt_missing")
    except FileNotFoundError:
        pass
    doc = repo.get_eval("rt_missing")
    assert doc is not None and doc["status"] == "failed"     # 后台异常被吞也不能把 running 挂着


def test_startup_recovers_zombie_evals(repo, policy):
    svc = make_service(repo, policy)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_many([
        {"run_id": "rt_old", "type": "routing", "status": "running", "created_at": now,
         "updated_at": now - timedelta(seconds=7200)},
        {"run_id": "rt_fresh", "type": "routing", "status": "running", "created_at": now,
         "updated_at": now - timedelta(seconds=5)},
    ])
    Services(svc.deps, svc, recover_stale=True)
    assert repo.get_eval("rt_old")["status"] == "failed"     # 重启后僵尸评测不再永远「进行中」
    assert repo.get_eval("rt_fresh")["status"] == "running"  # 新鲜的不动


def test_rsi_inheritance_excludes_eval_namespace(repo, policy):
    svc = make_service(repo, policy)
    now = datetime.now(timezone.utc)
    versions = {"lessons": 2, "routing": 1, "search": 1}
    for key in (CODE, f"evalns:{CODE}"):
        repo.db.dossiers.insert_one({"key": key, "ticker": CODE, "rsi_versions": versions,
                                     "lessons_used": ["L-001 r1"], "created_at": now})
    with TestClient(create_app(deps=svc.deps, checkpointer=None)) as c:
        inherit = c.get("/api/rsi/overview").json()["inheritance"]
    groups = [g for g in inherit if g["versions"] == versions]
    assert len(groups) == 1 and groups[0]["dossiers"] == 1   # 评测命名空间档案不算真实研究
