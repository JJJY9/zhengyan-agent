"""API 冒烟：三期 / 四期路由、作业看板、单页应用不越出 dist 目录。"""
from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient

from advisor.api.app import WEB_DIST, create_app
from advisor.contracts import ResearchRequest
from tests.fakes import CODE, as_of


@pytest.fixture
def client(service):
    service.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    with TestClient(create_app(deps=service.deps, checkpointer=None)) as c:
        yield c


def test_profile_and_fit(client):
    body = {"risk": {"max_drawdown_pct": 15, "horizon_months": 24}, "holdings": [{"ticker": CODE, "thesis": "上半年营业收入增长"}]}
    assert client.post("/api/profile", json=body).status_code == 409            # 未勾选确认
    assert client.post("/api/profile", json=body | {"confirmed": True}).json()["version"] == 1
    fit = client.get(f"/api/profile/fit/{CODE}").json()
    assert fit["holding"]["thesis"] == "上半年营业收入增长"
    assert {c["key"] for c in fit["fit"]["checks"]} == {"thesis", "drawdown", "horizon", "evidence", "freshness"}
    assert client.get("/api/profile/fit/000000").status_code == 404


def test_watch_routes(client):
    assert client.post("/api/watch/subscriptions", json={"ticker": "000001"}).status_code == 409   # 没有档案
    assert client.post("/api/watch/subscriptions", json={"ticker": CODE, "price_move_pct": 50}).json()["active"]
    assert "alert" in client.post(f"/api/watch/scan/{CODE}").json()
    ov = client.get("/api/watch").json()
    assert ov["subscriptions"][0]["ticker"] == CODE
    row = next(w for w in client.get("/api/watchlist").json() if w["key"] == CODE)
    assert row["subscribed"] is True and "open_alerts" in row


def test_strategy_routes(client):
    card = {"name": "测试策略", "thesis": "只纳入核验充分的档案", "universe": [CODE],
            "entry": [{"field": "verified", "op": ">=", "value": 3}], "change_reason": "API 冒烟测试新建策略卡"}
    v = client.post("/api/strategy", json=card).json()
    assert v["status"] == "draft"
    assert client.post(f"/api/strategy/{v['card_id']}/1/submit").status_code == 409
    assert client.post(f"/api/strategy/{v['card_id']}/1/paper").json()["status"] == "paper"
    job = client.post("/api/strategy/step", params={"day": date.today().isoformat()}).json()
    assert client.get(f"/api/jobs/{job['job_id']}").status_code == 200
    detail = client.get(f"/api/strategy/{v['card_id']}/1").json()
    assert detail["card"]["card_id"] == v["card_id"] and "stats" in detail["card"]
    assert "pe_pct" in client.get("/api/strategy").json()["fields"]


@pytest.mark.skipif(not WEB_DIST.exists(), reason="前端未构建")
def test_spa_does_not_escape_dist(client):
    r = client.get("/..%2F..%2Fpyproject.toml")
    assert "[project]" not in r.text
    assert client.get("/api/not-a-route").status_code == 404


def test_job_board_dedupes_running_key():
    """刷新页面后再点评测：同一候选还在跑时返回原作业，不并行跑第二次。"""
    from advisor.api.common import JobBoard

    class _Svc:
        def submit(self, fn):  # 不执行，作业保持 running
            pass

    board = JobBoard()
    a = board.start(_Svc(), "evaluate", lambda log: None, key="evaluate:r1")
    b = board.start(_Svc(), "evaluate", lambda log: None, key="evaluate:r1")
    assert a == b and board.running("evaluate:r1")["job_id"] == a["job_id"]
    assert board.start(_Svc(), "evaluate", lambda log: None, key="evaluate:r2") != a


def test_login_required_when_password_set(service):
    with TestClient(create_app(deps=service.deps, checkpointer=None, auth=("admin", "pw"))) as c:
        assert c.get("/api/session").json() == {"required": True, "logged_in": False}
        assert c.get("/api/meta").status_code == 401
        page = c.get("/rsi?x=1", follow_redirects=False)                     # 功能页未登录 → 跳登录页
        assert page.status_code == 303 and page.headers["location"] == "/login?next=/rsi%3Fx%3D1"
        assert c.post("/api/login", json={"user": "admin", "password": "wrong"}).status_code == 401
        assert c.post("/api/login", json={"user": "admin", "password": "pw"}).status_code == 200
        assert c.get("/api/meta").status_code == 200 and c.get("/api/session").json()["logged_in"]
        c.post("/api/logout")
        assert c.get("/api/meta").status_code == 401


def test_research_rejected_when_llm_not_configured(client):
    """没配大模型 Key：服务照常可浏览，但发起研究直接 400，而不是取完数才失败。"""
    from advisor.llm.client import DeepSeekClient
    from advisor.settings import Settings
    svc = client.app.state.services
    real = svc.deps.llm
    svc.deps.llm = DeepSeekClient(Settings(deepseek_api_key="", _env_file=None), roles={})
    try:
        r = client.post("/api/research", json={"question": f"复核 {CODE} 半年报业绩"})
        assert r.status_code == 400 and "DEEPSEEK_API_KEY" in r.json()["detail"]
        assert client.get("/api/dossiers").status_code == 200
    finally:
        svc.deps.llm = real


def test_missing_resources_return_404(client):
    assert client.post("/api/tasks/nope/resume", json={"approve": True}).status_code == 404
    assert client.post("/api/tasks/nope/cancel").status_code == 404
    assert client.post("/api/strategy/SC-999/1/paper").status_code == 404
    assert client.post("/api/strategy/SC-999/1/approve", json={"note": ""}).status_code == 404
    assert client.post("/api/strategy/SC-999/1/reject", json={"reason": "理由"}).status_code == 404
    assert client.post("/api/strategy/SC-999/rollback", json={"reason": "回滚理由不少于十个字"}).status_code == 404
    assert client.post("/api/rsi/runs/rsi_nope/approve", json={"note": ""}).status_code == 404
    assert client.post("/api/rsi/runs/rsi_nope/reject", json={"reason": "理由"}).status_code == 404
    assert client.post("/api/watch/alerts/al_nope/review").status_code == 404


def test_task_list_hides_eval_namespace(client, service):
    tid = service.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                        background=False, namespace="eval-test")
    assert service.deps.repo.get_task(tid) is not None
    ids = {t["task_id"] for t in client.get("/api/tasks").json()}
    assert tid not in ids                                   # 评测任务不出现在用户任务列表


def test_spa_fallback_survives_malformed_paths(client):
    assert client.get("/%00").status_code < 500             # 含 NUL 的路径回落首页，而不是 500
    assert client.get("/" + "a" * 5000).status_code < 500   # 超长路径同理


def test_job_board_prunes_finished_jobs(monkeypatch):
    from advisor.api.common import JobBoard

    class Sync:
        def submit(self, fn):
            fn()

    monkeypatch.setattr(JobBoard, "MAX_JOBS", 2)
    board = JobBoard()
    for _ in range(4):
        board.start(Sync(), "x", lambda log: 1)
    assert len(board.jobs) <= 2                             # 终态作业按插入顺序淘汰，内存有上限


def test_rsi_api_state_machine(client, service):
    """RSI 候选的 API 全链路：批准 → 发布（追加资产）→ 一键回滚，无需模型调用。"""
    from advisor.rsi.models import Run
    from advisor.services import Services

    svc = Services(service.deps, service)
    svc.rsi.ledger.bootstrap(svc.deps.policy)
    base = svc.rsi.ledger.active_asset("lessons")
    now = datetime.now(timezone.utc)
    run = Run(run_id="rsi_api_1", scenario="S1", kind="lessons", issue_ids=[], base_version=base.version,
              content={"cards": base.content["cards"]}, diff={},
              change_reason="API 冒烟：手工构造一个待审候选", direction="S1:class:smoke", status="review",
              created_at=now, updated_at=now)
    svc.rsi.ledger.insert_run(run)

    assert client.get("/api/rsi/overview").status_code == 200
    assert client.get("/api/rsi/runs").json()[0]["run_id"] == "rsi_api_1"
    assert client.post("/api/rsi/runs/rsi_api_1/approve", json={"note": ""}).json()["status"] == "approved"
    rel = client.post("/api/rsi/runs/rsi_api_1/release").json()
    assert rel["status"] == "released" and rel["released_version"] == 2
    assert svc.rsi.ledger.active_version("lessons") == 2
    rb = client.post("/api/rsi/rollback", json={"kind": "lessons", "reason": "API 冒烟测试回滚：理由够十个字"}).json()
    assert rb == {"kind": "lessons", "from": 2, "to": 1}
    assert client.post("/api/rsi/runs/rsi_api_1/release").status_code == 409    # 已回滚的候选不能再次发布


def test_export_all_dossiers_as_markdown_zip(client):
    import io
    import zipfile

    r = client.get("/api/dossiers/export.zip")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert names and any(CODE in n for n in names)
    assert "研究底稿" in z.read(names[0]).decode("utf-8")
