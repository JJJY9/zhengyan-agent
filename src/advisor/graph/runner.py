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
from advisor.harness.egress import EgressGuard
from advisor.tracing import span


class PoolSaturated(RuntimeError):
    """后台队列（排队+运行）达到上限：拒绝新作业而不是无限积压。"""


class ResearchService:
    MAX_PENDING = 64

    def __init__(self, deps: Deps, checkpointer, max_workers: int = 3, max_pending: int | None = None,
                 side_workers: int = 2):
        self.deps = deps
        self.graph = build_graph(deps, checkpointer)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research")
        self.max_pending = self.MAX_PENDING if max_pending is None else max_pending
        self._gate = threading.Lock()
        self._pending = 0
        # RSI/评测等后台作业的独立线程池与队列：长时间评测不能占满交互研究的 3 个线程与排队名额
        self._side_pool = ThreadPoolExecutor(max_workers=side_workers, thread_name_prefix="sidejob")
        self._side_gate = threading.Lock()
        self._side_pending = 0
        # 进程租约：回收僵尸任务只信租约（排队中的任务没有心跳，年龄判定会误杀）；
        # 与 deps 共用同一标识：评测等后台文档按它判定所属进程是否还活着
        self.instance = deps.instance
        self._futures: dict[Any, str] = {}          # 后台研究 future → task_id（停机收尾排队取消用）
        self._fut_lock = threading.Lock()
        self._hb_stop = threading.Event()
        threading.Thread(target=self._heartbeat, name="advisor-lease", daemon=True).start()
        self.guard = EgressGuard(
            deps.policy.egress.get("sensitive_patterns", []), deps.policy.egress.get("trade_intent_patterns", [])
        )

    def _heartbeat(self) -> None:
        while True:
            try:
                self.deps.repo.touch_lease(self.instance)
            except Exception:      # noqa: BLE001  续租失败不应打断服务
                pass
            if self._hb_stop.wait(60):
                return

    def shutdown(self) -> None:
        """服务停机：取消排队作业与任务、收尾本实例未完成的评测。

        心跳线程不在这里停：进程可能仍在排空在跑任务，停租约会让并行启动的新实例
        把它们当无主任务标失败；守护线程随进程退出即消失，租约随后自然过期。
        """
        for pool in (self._side_pool, self._pool):
            pool.shutdown(wait=False, cancel_futures=True)
        try:
            self.deps.repo.cancel_queued_tasks(self.instance, "服务停止：排队中的任务已取消")
            self.deps.repo.fail_instance_evals(self.instance, "服务停止，评测中断；可重新发起")
            with self._fut_lock:
                cancelled = [tid for f, tid in self._futures.items() if f.cancelled()]
                self._futures.clear()
            for task_id in cancelled:   # 已批准但排队即被取消：_run 永不执行，明确收尾防止悬挂
                self.deps.repo.finish_task(task_id, "cancelled",
                                           error="服务停止：已批准的恢复未执行，可重新发起")
        except Exception:      # noqa: BLE001
            pass

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
              review_focus: list[str] | None = None, admission: str = "interactive") -> str:
        if getattr(self.deps.llm, "configured", True) is False:   # 没配 Key 就不开工，免得白白取数后才失败
            from advisor.llm import LLMNotConfigured
            raise LLMNotConfigured("未配置 DEEPSEEK_API_KEY，不能发起新研究：请在 .env 中填写后重启服务（已有档案仍可浏览）")
        if admission != "interactive" and background:
            raise ValueError("admission='side' 只支持前台执行：评测用例在侧栏线程内联运行")
        guarded = admission == "interactive"
        if guarded:
            self._acquire_or_raise()
        try:
            task_id = "t_" + uuid.uuid4().hex[:10]
            # 任务请求会长期留在任务列表/导出里：含个人信息的问句按 consent 承诺脱敏后落盘
            request_doc = req.model_dump(mode="json")
            request_doc["question"] = self.guard.redact_question(req.question)
            self.deps.repo.create_task(task_id, request_doc | ({"namespace": namespace} if namespace else {}),
                                       instance=self.instance)
            payload = {
                "task_id": task_id, "question": req.question, "ticker": req.ticker, "namespace": namespace or "",
                "as_of": (req.as_of or date.today()).isoformat(), "route_requested": req.route,
                "review_focus": review_focus or [],
            }
            self._submit(task_id, payload, background, guarded=guarded)
        except Exception:
            if guarded:
                self._release()      # 提交没成功：把名额还回去
            raise
        return task_id

    def resume(self, task_id: str, approve: bool, background: bool = True,
               admission: str = "interactive") -> None:
        task = self.deps.repo.get_task(task_id)
        if not task or task["status"] != "waiting_approval":
            raise ValueError("任务不在等待确认状态")
        if admission != "interactive" and background:
            raise ValueError("admission='side' 只支持前台执行：评测用例在侧栏线程内联运行")
        ticket = task.get("interrupt") or {}
        reason = ""
        if ticket.get("ttl_s") and time.time() - ticket.get("issued_at", 0) > ticket["ttl_s"]:
            approve, reason = False, "审批已过期"
        guarded = admission == "interactive"
        if guarded:
            self._acquire_or_raise()
        try:
            # 原子夺取「待审批 → 运行中」：并发双击/客户端重试只有一次真正恢复，
            # 否则同一 thread 会被提交两次 Command(resume)，主席与提交节点重放、生成两份档案
            if not self.deps.repo.try_resume_task(task_id, instance=self.instance):
                raise ValueError("任务已被另一次恢复操作接管")
            # 审批等待（含停机）不计入执行时限：暂停时长记进任务账本，预算检查按开始时间扣除；
            # 不对暂停中的图做 update_state——那会重写检查点、破坏中断的 resume 重放顺序
            issued = float(ticket.get("issued_at") or 0)
            if issued:
                self.deps.repo.add_paused_time(task_id, max(time.time() - issued, 0.0))
            self.deps.repo.push_event(task_id, "approval", f"{'同意' if approve else '拒绝'}{('（' + reason + '）') if reason else ''}")
            decision = {**ticket, "approve": approve, "reason": reason}
            self._submit(task_id, Command(resume=self._resume_payload(task_id, ticket, decision)), background,
                         guarded=guarded)
        except Exception:
            if guarded:
                self._release()      # 提交没成功：把名额还回去
            raise

    def _resume_payload(self, task_id: str, ticket: dict, decision: dict) -> dict:
        """恢复值按 interrupt id 定向投递。

        深研多个专家同时停在审批上时，裸值恢复会被已重放的错误任务抢走（甚至直接报错要求按 id 恢复）；
        以当前检查点里的待审批中断为准定向，票据里的 id 过期时退回按参数指纹匹配、再退回唯一待审批。
        """
        config = {"configurable": {"thread_id": task_id}, "recursion_limit": 60}
        try:
            pending = [i for t in self.graph.get_state(config).tasks for i in t.interrupts]
        except Exception:
            pending = []
        target = next((i for i in pending if i.id == ticket.get("interrupt_id")), None)
        if target is None and ticket.get("args_hash"):
            target = next((i for i in pending
                           if isinstance(i.value, dict) and i.value.get("args_hash") == ticket.get("args_hash")), None)
        if target is None and len(pending) == 1:
            target = pending[0]
        return {target.id: decision} if target is not None else decision

    def cancel(self, task_id: str) -> str | None:
        status = self.deps.repo.request_cancel(task_id)
        if status:
            self.deps.repo.push_event(task_id, "runner", "收到取消请求：撤销授权，停止在途任务")
        return status

    def submit(self, fn, *args):
        """在研究线程池里执行后台作业；队列满时拒绝而不是无限积压。"""
        self._acquire_or_raise()
        try:
            return self._pool.submit(self._guarded, fn, *args)
        except RuntimeError:
            self._release()
            raise

    def submit_side(self, fn, *args):
        """侧栏作业（RSI 提议/评测、订阅扫描、模拟盘推进）走独立线程池：
        再长的评测也只占自己的名额，交互研究照常可以发起（取消通道仍待做，见迭代记录）。"""
        with self._side_gate:
            if self._side_pending >= self.max_pending:
                raise PoolSaturated(f"后台作业队列已满（最多 {self.max_pending} 个排队/运行），请稍后再试")
            self._side_pending += 1
        try:
            return self._side_pool.submit(self._guard_side, fn, *args)
        except RuntimeError:
            with self._side_gate:
                self._side_pending = max(self._side_pending - 1, 0)
            raise

    def _guard_side(self, fn, *args):
        try:
            fn(*args)
        finally:
            with self._side_gate:
                self._side_pending = max(self._side_pending - 1, 0)

    def _submit(self, task_id: str, payload: Any, background: bool, guarded: bool = True) -> None:
        """guarded=True（交互研究）时调用前必须已 _acquire_or_raise()；评测用例（admission="side"）
        不占交互名额、由侧栏作业自身的 worker 串行执行，只走裸 _run。"""
        if not guarded:
            if background:
                self._pool.submit(self._run, task_id, payload)
            else:
                self._run(task_id, payload)
            return
        if background:
            fut = self._pool.submit(self._guarded, self._run, task_id, payload)
            with self._fut_lock:
                self._futures[fut] = task_id
            fut.add_done_callback(self._forget_future)
        else:
            self._guarded(self._run, task_id, payload)

    def _forget_future(self, fut) -> None:
        if fut.cancelled():
            return                     # 停机时排队即被取消：留给 shutdown 统一收尾
        with self._fut_lock:
            self._futures.pop(fut, None)

    def _run(self, task_id: str, payload: Any) -> None:
        repo = self.deps.repo
        config = {"configurable": {"thread_id": task_id}, "recursion_limit": 60}
        if not repo.mark_running(task_id, instance=self.instance):
            return                     # 已被回收/取消：绝不复活执行
        repo.touch_lease(self.instance)
        try:
            with span("research", task_id=task_id):
                result = self.graph.invoke(payload, config)
            snapshot = self.graph.get_state(config)
            pending = [i for t in snapshot.tasks for i in t.interrupts]
            if pending:
                # 存下 interrupt id：恢复时按 id 定向投递，深研多专家并行待审批时裸值会送错任务
                if not repo.finish_task(task_id, "waiting_approval", instance=self.instance,
                                        repair_recovered=True,
                                        interrupt={**pending[0].value, "issued_at": time.time(),
                                                   "interrupt_id": pending[0].id}):
                    repo.push_event(task_id, "runner", "任务已被回收/取消，忽略等待审批状态写入")
                return
            status = result.get("status", "done")
            if not repo.finish_task(
                task_id, status, instance=self.instance, repair_recovered=True,
                error=result.get("error"),
                result={"dossier_key": result.get("dossier_key"), "version": result.get("version"),
                        "quality": result.get("quality"), "budget": result.get("budget"), "route": result.get("route"),
                        "routing": result.get("routing")},
            ):
                repo.push_event(task_id, "runner", "任务已被回收/取消，忽略收尾状态写入（拒绝复活）")
        except TaskCancelled as e:
            repo.push_event(task_id, "runner", str(e))
            repo.finish_task(task_id, "cancelled", instance=self.instance, repair_recovered=True)
        except Exception as e:  # 失败可见，不吞错误
            repo.push_event(task_id, "runner", f"异常：{type(e).__name__}: {e}")
            repo.finish_task(task_id, "failed", instance=self.instance, repair_recovered=True,
                             error=f"{type(e).__name__}: {e}")
