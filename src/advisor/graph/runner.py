"""任务运行器：后台线程执行图，处理人在环中断、恢复与取消。中断恢复只恢复工作状态，不恢复过期权限。"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from langgraph.types import Command

from advisor.contracts import ResearchRequest
from advisor.graph.builder import build_graph
from advisor.graph.deps import Deps
from advisor.harness import TaskCancelled
from advisor.tracing import span


class PoolSaturated(RuntimeError):
    """后台队列（排队+运行）达到上限：拒绝新作业而不是无限积压。"""


class ResearchService:
    MAX_PENDING = 64

    def __init__(self, deps: Deps, checkpointer, max_workers: int = 3, max_pending: int | None = None):
        self.deps = deps
        self.graph = build_graph(deps, checkpointer)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research")
        self.max_pending = self.MAX_PENDING if max_pending is None else max_pending
        self._gate = threading.Lock()
        self._pending = 0

    def _acquire_or_raise(self) -> None:
        with self._gate:
            if self._pending >= self.max_pending:
                raise PoolSaturated(f"后台任务队列已满（最多 {self.max_pending} 个排队/运行），请稍后再试")
            self._pending += 1

    def _release(self) -> None:
        with self._gate:
            self._pending = max(self._pending - 1, 0)

    def _guarded(self, fn, *args):
        try:
            fn(*args)
        finally:
            self._release()

    def start(self, req: ResearchRequest, background: bool = True, namespace: str | None = None,
              review_focus: list[str] | None = None) -> str:
        if getattr(self.deps.llm, "configured", True) is False:   # 没配 Key 就不开工，免得白白取数后才失败
            from advisor.llm import LLMNotConfigured
            raise LLMNotConfigured("未配置 DEEPSEEK_API_KEY，不能发起新研究：请在 .env 中填写后重启服务（已有档案仍可浏览）")
        self._acquire_or_raise()
        try:
            task_id = "t_" + uuid.uuid4().hex[:10]
            self.deps.repo.create_task(task_id, req.model_dump(mode="json") | ({"namespace": namespace} if namespace else {}))
            payload = {
                "task_id": task_id, "question": req.question, "ticker": req.ticker, "namespace": namespace or "",
                "as_of": (req.as_of or date.today()).isoformat(), "route_requested": req.route,
                "review_focus": review_focus or [],
            }
            self._submit(task_id, payload, background)
        except Exception:
            self._release()          # 提交没成功：把名额还回去
            raise
        return task_id

    def resume(self, task_id: str, approve: bool, background: bool = True) -> None:
        task = self.deps.repo.get_task(task_id)
        if not task or task["status"] != "waiting_approval":
            raise ValueError("任务不在等待确认状态")
        ticket = task.get("interrupt") or {}
        reason = ""
        if ticket.get("ttl_s") and time.time() - ticket.get("issued_at", 0) > ticket["ttl_s"]:
            approve, reason = False, "审批已过期"
        self._acquire_or_raise()
        try:
            # 原子夺取「待审批 → 运行中」：并发双击/客户端重试只有一次真正恢复，
            # 否则同一 thread 会被提交两次 Command(resume)，主席与提交节点重放、生成两份档案
            if not self.deps.repo.try_resume_task(task_id):
                raise ValueError("任务已被另一次恢复操作接管")
            # 审批等待（含停机）不计入执行时限：把预算时钟顺延暂停时长
            issued = float(ticket.get("issued_at") or 0)
            if issued:
                self._rebase_paused_clock(task_id, max(time.time() - issued, 0.0))
            self.deps.repo.push_event(task_id, "approval", f"{'同意' if approve else '拒绝'}{('（' + reason + '）') if reason else ''}")
            self._submit(task_id, Command(resume={**ticket, "approve": approve, "reason": reason}), background)
        except Exception:
            self._release()          # 提交没成功：把名额还回去
            raise

    def _rebase_paused_clock(self, task_id: str, pause_s: float) -> None:
        """把检查点里的预算 started_at 顺延暂停时长：恢复后的时限只统计实际执行时间。"""
        if pause_s <= 0:
            return
        config = {"configurable": {"thread_id": task_id}, "recursion_limit": 60}
        try:
            budget = (self.graph.get_state(config).values or {}).get("budget")
            if not budget:
                return
            budget = dict(budget)
            budget["started_at"] = float(budget.get("started_at", time.time())) + pause_s
            self.graph.update_state(config, {"budget": budget})
        except Exception as e:  # 顺延失败不应阻止恢复，但要记录
            self.deps.repo.push_event(task_id, "runner", f"预算时钟顺延失败：{type(e).__name__}: {e}")

    def cancel(self, task_id: str) -> str | None:
        status = self.deps.repo.request_cancel(task_id)
        if status:
            self.deps.repo.push_event(task_id, "runner", "收到取消请求：撤销授权，停止在途任务")
        return status

    def submit(self, fn, *args):
        """在研究线程池里执行后台作业（评测、RSI、订阅扫描等）；队列满时拒绝而不是无限积压。"""
        self._acquire_or_raise()
        try:
            return self._pool.submit(self._guarded, fn, *args)
        except RuntimeError:
            self._release()
            raise

    def _submit(self, task_id: str, payload: Any, background: bool) -> None:
        """调用前必须已 _acquire_or_raise()；名额由 _guarded 在作业结束时释放。"""
        if background:
            self._pool.submit(self._guarded, self._run, task_id, payload)
        else:
            self._guarded(self._run, task_id, payload)

    def _run(self, task_id: str, payload: Any) -> None:
        repo = self.deps.repo
        config = {"configurable": {"thread_id": task_id}, "recursion_limit": 60}
        repo.update_task(task_id, status="running")
        try:
            with span("research", task_id=task_id):
                result = self.graph.invoke(payload, config)
            snapshot = self.graph.get_state(config)
            pending = [i for t in snapshot.tasks for i in t.interrupts]
            if pending:
                repo.update_task(task_id, status="waiting_approval", interrupt={**pending[0].value, "issued_at": time.time()})
                return
            status = result.get("status", "done")
            repo.update_task(
                task_id, status=status, error=result.get("error"),
                result={"dossier_key": result.get("dossier_key"), "version": result.get("version"),
                        "quality": result.get("quality"), "budget": result.get("budget"), "route": result.get("route"),
                        "routing": result.get("routing")},
            )
        except TaskCancelled as e:
            repo.push_event(task_id, "runner", str(e))
            repo.update_task(task_id, status="cancelled")
        except Exception as e:  # 失败可见，不吞错误
            repo.push_event(task_id, "runner", f"异常：{type(e).__name__}: {e}")
            repo.update_task(task_id, status="failed", error=f"{type(e).__name__}: {e}")
