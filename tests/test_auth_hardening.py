"""认证与请求边界加固：随机会话、登出吊销、登录限速、请求体上限、评测参数上限、后台队列上限。"""
from fastapi.testclient import TestClient

from advisor.api.app import _LoginThrottle, _SessionStore, create_app

ADMIN = ("admin", "pw")


def _client(service, **kw):
    return TestClient(create_app(deps=service.deps, checkpointer=None, auth=ADMIN, **kw))


def _login(c, password="pw"):
    return c.post("/api/login", json={"user": "admin", "password": password})


def test_sessions_are_random_and_revocable(service):
    with _client(service) as c:
        assert _login(c).status_code == 200
        first = c.cookies.get("advisor_session")
        assert c.get("/api/meta").status_code == 200
        c.post("/api/logout")
        assert c.get("/api/meta").status_code == 401
        replay = c.get("/api/meta", headers={"cookie": f"advisor_session={first}"})
        assert replay.status_code == 401                    # 登出后旧 token 已吊销，重放无效
        assert _login(c).status_code == 200
        assert c.cookies.get("advisor_session") != first    # 每次登录都是新的随机 token


def test_logout_requires_session(service):
    with _client(service) as c:
        assert c.post("/api/logout").status_code == 401     # 不再允许跨站匿名登出


def test_login_throttled_after_failures(service):
    with _client(service) as c:
        for _ in range(10):
            assert _login(c, "wrong").status_code == 401
        r = _login(c)                                        # 正确口令也被临时拒绝
        assert r.status_code == 429 and r.headers.get("retry-after")


def test_session_store_expiry_and_cap():
    now = [1000.0]
    store = _SessionStore(ttl_s=10, max_sessions=2, now=lambda: now[0])
    a = store.create()
    assert store.valid(a)
    now[0] += 11
    assert not store.valid(a)
    tokens = [store.create() for _ in range(3)]
    assert len(store) == 2 and store.valid(tokens[-1])


def test_login_throttle_window():
    now = [0.0]
    t = _LoginThrottle(limit=2, window_s=60, now=lambda: now[0])
    t.fail("ip")
    t.fail("ip")
    assert t.blocked("ip")
    now[0] = 61
    assert not t.blocked("ip")


def test_large_body_rejected_before_auth(service):
    with _client(service) as c:
        big = "a" * (2 * 1024 * 1024 + 10)
        r = c.post("/api/login", json={"user": "admin", "password": big})
        assert r.status_code == 413


def test_login_fields_are_bounded(service):
    with _client(service) as c:
        assert c.post("/api/login", json={"user": "admin", "password": "x" * 300}).status_code == 422


def test_cookie_secure_setting(service):
    with _client(service, cookie_secure=True) as c:
        r = _login(c)
        assert r.status_code == 200 and "secure" in r.headers["set-cookie"].lower()


def test_compare_rejects_oversized_lists(service):
    with TestClient(create_app(deps=service.deps, checkpointer=None)) as c:
        assert c.post("/api/evals/compare", json={"routes": ["standard"] * 4}).status_code == 422
        assert c.post("/api/evals/compare", json={"case_ids": ["x"] * 201}).status_code == 422


def test_compare_dedupes_routes(service, monkeypatch):
    from advisor.evals.compare import load_cases

    captured = {}

    def fake_submit(fn, *args):
        captured["args"] = args

    case_id = load_cases()[0]["id"]
    with TestClient(create_app(deps=service.deps, checkpointer=None)) as c:
        monkeypatch.setattr(c.app.state.services, "submit", fake_submit)
        r = c.post("/api/evals/compare", json={"routes": ["standard", "standard"], "case_ids": [case_id]})
        assert r.status_code == 200
    assert captured["args"][2] == ["standard"]              # 重复 route 已去重，不会放大后台研究次数


def test_saturated_pool_returns_429(service, monkeypatch):
    with TestClient(create_app(deps=service.deps, checkpointer=None)) as c:
        monkeypatch.setattr(c.app.state.services.research, "max_pending", 0)
        assert c.post("/api/research", json={"question": "复核 600999 半年报业绩"}).status_code == 429
        assert c.post("/api/evals/routing").status_code == 429
        evals = c.get("/api/evals").json()
        assert any(e.get("status") == "failed" and "队列" in (e.get("error") or "") for e in evals)
