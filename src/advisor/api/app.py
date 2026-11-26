"""FastAPI 服务：组装各业务路由，并在构建过前端时提供 Web 界面。

路由按业务拆分：research（研究任务）· dossiers（档案与证据）· evals（评测）· rsi（自进化）
· personal（用户确认档案、个人情境官）· watch（订阅与增量复核）· strategy（策略卡与模拟盘）
"""
import json
import secrets
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from advisor.api import dossiers, evals, personal, research, rsi, strategy, watch
from advisor.api.common import JobBoard, Jobs
from advisor.graph.deps import Deps
from advisor.scheduler import Scheduler
from advisor.services import build_services
from advisor.settings import ROOT, get_settings

WEB_DIST = ROOT / "web" / "dist"
ROUTERS = (research, dossiers, evals, rsi, personal, watch, strategy)


def create_app(deps: Deps | None = None, checkpointer=None, auth: tuple[str, str] | None = None,
               cookie_secure: bool = False) -> FastAPI:
    production = deps is None
    if production and auth is None and (cfg := get_settings()).auth_password:
        auth = (cfg.auth_user, cfg.auth_password)
        cookie_secure = cookie_secure or cfg.auth_cookie_secure

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = build_services(deps, checkpointer, recover_stale=True)
        app.state.services = services
        app.state.jobs = JobBoard()
        hours = float(services.deps.policy.watch.get("scan_interval_hours", 0)) if production else 0
        scheduler = Scheduler(services, hours).start()
        yield
        scheduler.stop()
        services.research.shutdown()          # 取消排队作业并把注定不执行的任务标取消

    app = FastAPI(title="股票投资顾问 Agent", version="0.4.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(_BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    if auth:
        _require_login(app, *auth, cookie_secure=cookie_secure)
    else:
        @app.get("/api/session")
        def session():
            return {"required": False, "logged_in": True}
    for module in ROUTERS:
        app.include_router(module.router)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, jobs: Jobs):
        return jobs.cancel(job_id)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, jobs: Jobs):
        return jobs.get(job_id)

    if WEB_DIST.exists():
        _mount_web(app)
    return app


SESSION_COOKIE = "advisor_session"
SESSION_DAYS = 7
PUBLIC_PATHS = {"/", "/login", "/api/login", "/api/session"}
MAX_BODY_BYTES = 2 * 1024 * 1024
LOGIN_MAX_FAILURES = 10
LOGIN_WINDOW_S = 300


class _SessionStore:
    """进程内会话表：随机 token、服务端过期、登出即吊销。重启服务全部失效（需重新登录）。"""

    def __init__(self, ttl_s: float, max_sessions: int = 100, now=time.monotonic):
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._now = now
        self._expires: dict[str, float] = {}
        self._lock = threading.Lock()          # 检查-淘汰必须原子：并发建会话时多个线程会选中同一个待淘汰 token

    def create(self) -> str:
        with self._lock:
            self._prune()
            token = secrets.token_urlsafe(32)
            self._expires[token] = self._now() + self.ttl_s
            if len(self._expires) > self.max_sessions:          # 超过上限先淘汰最早到期的会话
                self._expires.pop(min(self._expires, key=self._expires.get), None)
            return token

    def valid(self, token: str | None) -> bool:
        with self._lock:
            if not token or token not in self._expires:
                return False
            if self._expires[token] <= self._now():
                self._expires.pop(token, None)
                return False
            return True

    def revoke(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._expires.pop(token, None)

    def __len__(self) -> int:
        with self._lock:
            self._prune()
            return len(self._expires)

    def _prune(self) -> None:
        now = self._now()
        for token, exp in list(self._expires.items()):
            if exp <= now:
                self._expires.pop(token, None)


class _LoginThrottle:
    """登录失败限速：按 (IP, 账号) 分桶，另留按 IP 的总闸；过期桶定期清理。

    只按账号分桶会丢 IP 级限速：同一个 IP 换着用户名喷洒，每个账号都拿到满额尝试次数；
    只按 IP 分桶又会因反代把全员算成同一来源而一锁全锁——两级并用。
    """

    IP_LIMIT_FACTOR = 5      # 单 IP 在窗口内的失败上限 = 账号上限 × 该系数
    PRUNE_AT = 10000         # 失败桶数量超过此时全量清理过期键，避免无界增长

    def __init__(self, limit: int = LOGIN_MAX_FAILURES, window_s: float = LOGIN_WINDOW_S, now=time.monotonic):
        self.limit, self.window_s, self._now = limit, window_s, now
        self._fails: dict[tuple, list[float]] = {}

    def _recent(self, key: tuple) -> list[float]:
        cutoff = self._now() - self.window_s
        fails = [t for t in self._fails.get(key, []) if t > cutoff]
        if fails:
            self._fails[key] = fails
        else:
            self._fails.pop(key, None)
        return fails

    def blocked(self, ip: str, user: str) -> bool:
        if len(self._recent(("i", ip))) >= self.limit * self.IP_LIMIT_FACTOR:
            return True
        return len(self._recent(("a", ip, user))) >= self.limit

    def fail(self, ip: str, user: str) -> None:
        self._recent(("a", ip, user))
        self._fails.setdefault(("a", ip, user), []).append(self._now())
        self._fails.setdefault(("i", ip), []).append(self._now())
        if len(self._fails) > self.PRUNE_AT:
            self._prune_all()

    def success(self, ip: str, user: str) -> None:
        self._fails.pop(("a", ip, user), None)

    def _prune_all(self) -> None:
        cutoff = self._now() - self.window_s
        for key in list(self._fails):
            fails = [t for t in self._fails[key] if t > cutoff]
            if fails:
                self._fails[key] = fails
            else:
                self._fails.pop(key, None)
        if len(self._fails) > self.PRUNE_AT:
            # 窗口内仍超上限（喷洒制造海量新账号桶）：按最近一次失败时间淘汰最旧的账号桶；
            # IP 总闸桶保留——喷的人越多，这个 IP 越该被挡
            acct = sorted((max(v), k) for k, v in self._fails.items() if k[0] == "a")
            for _, key in acct[: len(self._fails) - self.PRUNE_AT]:
                self._fails.pop(key, None)


class _BodyTooLarge(Exception):
    pass


class _BodyLimitMiddleware:
    """请求体上限（默认 2MB）：按 Content-Length 预检、流式累计兜底；超过即 413，未登录请求也撑不爆内存。"""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        length = headers.get(b"content-length")
        if length is not None:
            try:
                too_large = int(length) > self.max_bytes
            except ValueError:
                return await _reply(send, 400, "Content-Length 无效")
            if too_large:
                return await _reply(send, 413, _too_large_detail())
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _reply(send, 413, _too_large_detail())


def _too_large_detail() -> str:
    return f"请求体超过 {MAX_BODY_BYTES // (1024 * 1024)}MB 上限"


async def _reply(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}, ensure_ascii=False).encode()
    try:
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json; charset=utf-8"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
    except RuntimeError:            # 响应已开始（流式场景）时无法再改状态码，忽略
        pass


class LoginBody(BaseModel):
    user: str = Field(max_length=128)
    password: str = Field(max_length=256)


def _require_login(app: FastAPI, user: str, password: str, cookie_secure: bool = False) -> None:
    """访问账号密码：首页和登录页公开，其余页面与接口须先登录（随机会话 cookie，登出即吊销）。"""
    sessions = _SessionStore(SESSION_DAYS * 86400)
    throttle = _LoginThrottle()

    def logged_in(request: Request) -> bool:
        return sessions.valid(request.cookies.get(SESSION_COOKIE))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/assets/") or logged_in(request):
            return await call_next(request)
        if path.startswith("/api/"):
            return JSONResponse({"detail": "请先登录"}, status_code=401)
        target = path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target)}", status_code=303)

    @app.post("/api/login")
    def login(body: LoginBody, request: Request):
        ip = request.client.host if request.client else "unknown"
        name = body.user.strip()
        if throttle.blocked(ip, name):         # (IP,账号) 分桶 + IP 总闸：喷洒挡得住，单账号爆破锁不住别人
            raise HTTPException(429, "登录尝试过多，请稍后再试", headers={"Retry-After": str(LOGIN_WINDOW_S)})
        user_ok = secrets.compare_digest(body.user.encode(), user.encode())
        password_ok = secrets.compare_digest(body.password.encode(), password.encode())
        if not (user_ok and password_ok):
            throttle.fail(ip, name)
            raise HTTPException(401, "账号或密码不对")
        throttle.success(ip, name)
        resp = JSONResponse({"ok": True})
        https = cookie_secure or request.url.scheme == "https" \
            or request.headers.get("x-forwarded-proto", "").strip().lower() == "https"
        resp.set_cookie(SESSION_COOKIE, sessions.create(), max_age=SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=https)
        return resp

    @app.post("/api/logout")
    def logout(request: Request):
        sessions.revoke(request.cookies.get(SESSION_COOKIE))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/api/session")
    def session(request: Request):
        return {"required": True, "logged_in": logged_in(request)}


def _mount_web(app: FastAPI) -> None:
    """单页应用：静态资源直接返回，其余路径回落到 index.html。只允许读取 dist 目录内的文件。"""
    root = WEB_DIST.resolve()
    app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "接口不存在")
        file = None
        try:
            file = (root / path).resolve()
            ok = bool(path) and file.is_file() and file.is_relative_to(root)
        except (ValueError, OSError):     # 含 NUL、超长路径等非法输入：回落首页，而不是 500
            ok = False
        return FileResponse(file if ok and file is not None else root / "index.html")
