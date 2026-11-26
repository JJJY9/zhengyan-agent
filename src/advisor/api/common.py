"""各路由共用：服务容器注入、后台作业看板、业务错误 → HTTP 状态码。"""
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Request

from advisor.graph.runner import PoolSaturated
from advisor.harness import TaskCancelled
from advisor.services import Services

OWNER = "owner"            # 单人阶段的操作人；审计里如实标出「自批」


def get_services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(get_services)]


class JobBoard:
    """耗时操作（RSI 提议/评测、订阅扫描、模拟盘推进）在后台线程执行，前端轮询日志。

    并发安全：start/running/get 都用锁保护（FastAPI 同步端点在线程池里真并发）；
    终态作业按插入顺序淘汰，只保留最近 MAX_JOBS 条，进程内存不会随使用时长线性增长。
    """

    MAX_JOBS = 200

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _running_locked(self, key: str) -> dict | None:
        return next((j for j in self.jobs.values() if j.get("key") == key and j["status"] == "running"), None)

    def _prune_locked(self) -> None:
        overflow = len(self.jobs) - self.MAX_JOBS
        if overflow <= 0:
            return
        for k in [k for k, j in list(self.jobs.items()) if j["status"] != "running"][:overflow]:
            self.jobs.pop(k, None)

    def start(self, svc: Services, kind: str,
              fn: Callable[[Callable[[str], None], Callable[[], bool]], object],
              key: str | None = None) -> dict:
        """key 相同的作业还在跑时不重复启动，直接返回那一个（刷新页面后重复点击不会并行跑两次）。

        fn 收到 (log, is_cancelled)：长作业（RSI 评测、扫描、推盘）应在每个重复/子任务边界检查
        is_cancelled() 并抛 TaskCancelled，作业以「已取消」收尾而不是被硬杀。"""
        with self._lock:
            if key and (existing := self._running_locked(key)):
                return {"job_id": existing["job_id"]}
            jid = "job_" + uuid.uuid4().hex[:12]
            job = {"job_id": jid, "kind": kind, "key": key, "status": "running", "log": [], "result": None,
                   "error": None, "cancel": False, "started_at": datetime.now(timezone.utc)}
            self.jobs[jid] = job
            self._prune_locked()

        def run():
            try:
                out = fn(job["log"].append, lambda: job.get("cancel", False))
                job.update(status="done", result=getattr(out, "run_id", None) or out)
            except TaskCancelled as e:
                job.update(status="cancelled", error=str(e) or "作业已取消")
            except Exception as e:      # 失败可见
                job.update(status="failed", error=f"{type(e).__name__}: {e}")
        try:
            svc.submit(run)
        except PoolSaturated as e:      # 队列满：不登记作业，直接 429
            with self._lock:
                self.jobs.pop(jid, None)
            raise HTTPException(429, str(e)) from e
        return {"job_id": jid}

    def cancel(self, job_id: str) -> dict:
        """请求取消：置标志后由作业在下一个协作点以 TaskCancelled 结束（无法硬杀线程）。"""
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "作业不存在")
            if job["status"] != "running":
                raise HTTPException(409, "作业已结束，无需取消")
            job["cancel"] = True
            if any("取消请求" in ln for ln in job["log"]):
                raise HTTPException(409, "已请求取消，等待作业在下一个检查点停止")
            job["log"].append("已收到取消请求，作业将在下一个检查点停止")
        return {"ok": True}

    def running(self, key: str) -> dict | None:
        with self._lock:
            return self._running_locked(key)

    def get(self, job_id: str) -> dict:
        with self._lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "作业不存在")
        return job


def get_jobs(request: Request) -> JobBoard:
    return request.app.state.jobs


Jobs = Annotated[JobBoard, Depends(get_jobs)]


@contextmanager
def conflict_on(*errors: type[Exception]):
    """状态机拒绝的操作返回 409，带上原因。"""
    try:
        yield
    except errors as e:
        raise HTTPException(409, str(e)) from e
