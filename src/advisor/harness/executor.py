"""Harness：所有 Skill 与模型调用的唯一入口。

顺序：撤销检查 → 注册检查 → 流程白名单 → 输入校验 → 前置检查（时点/出域）→ 审批门控
→ 预算 → 幂等缓存 → 超时执行与有限重试 → 输出校验 → 审计（含 trace_id、节点）。
任一安全关键检查异常时默认阻断。
"""
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, ValidationError

from advisor.harness.budget import Budget
from advisor.harness.egress import EgressGuard
from advisor.harness.errors import ApprovalRequired, BudgetExceeded, HarnessDenied, SkillFailed, TaskCancelled
from advisor.harness.policy import Policy
from advisor.tracing import span

MAX_RETRIES = 2
MIN_LLM_TOKENS = 32          # 剩余预算不够产出最小合法 JSON 时不再发起必败的模型调用
LLM_TIMEOUT_CAP_S = 300.0


class _SkillTimeout(Exception):
    """单次调用的等待超时（区别于技能自身抛出的 TimeoutError）。"""


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """在独立守护线程里执行单次调用，超时只放弃等待。

    不用共享线程池：Python 无法终止跑飞的线程，固定 16 线程的池子在多次超时后会被孤儿任务
    占满，之后所有技能调用（包括纯本地的 compute.*）都会排队到超时，非守护池线程还会阻止
    进程退出。独立守护线程最多留下一个孤儿，不拖垮其他调用，也不阻塞退出。
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["out"] = fn()
        except BaseException as e:  # noqa: BLE001  原样带回调用线程
            box["err"] = e
        finally:
            done.set()

    threading.Thread(target=target, name="harness-skill", daemon=True).start()
    if not done.wait(timeout_s):
        raise _SkillTimeout(f"调用超时（{timeout_s}s）")
    if "err" in box:
        raise box["err"]
    return box["out"]


@dataclass
class RunContext:
    task_id: str
    route: str
    as_of: date
    principal: str = "researcher"
    node: str = ""
    is_cancelled: Callable[[], bool] = field(default=lambda: False)


class Harness:
    def __init__(
        self,
        policy: Policy,
        registry: Any,
        audit: Callable[[dict], None],
        budget: Budget,
        ctx: RunContext,
        llm: Any = None,
        decision: Any = None,
        cache: dict | None = None,
        flight: dict | None = None,
        charge_sink: Callable[[str, float], None] | None = None,
    ):
        self.policy = policy
        self.registry = registry
        self.audit = audit
        self.budget = budget
        self.ctx = ctx
        self.llm_client = llm
        self.decision_client = decision
        self._charge_sink = charge_sink
        self.egress = EgressGuard(
            policy.egress.get("sensitive_patterns", []),
            policy.egress.get("trade_intent_patterns", []),
        )
        # 幂等缓存：研究任务传入"任务级"字典（同一 task_id 的所有节点、并行专家共用）；不传则只在本实例内有效
        self._cache: dict[str, BaseModel] = cache if cache is not None else {}
        # 在途占位（与 _cache 同生命周期）：并发的同参调用只执行一次，其余等待复用结果
        self._flight: dict[str, threading.Event] = flight if flight is not None else {}

    def _charge(self, kind: str, amount: float = 1) -> None:
        """本地计费 + 任务级持久账本：节点被审批中断不会返回，账目不能只活在节点局部对象里。"""
        self.budget.charge(kind, amount)
        if self._charge_sink is not None:
            self._charge_sink(kind, amount)

    def _base_record(self, kind: str, name: str) -> dict:
        return {"task_id": self.ctx.task_id, "kind": kind, "name": name, "route": self.ctx.route,
                "node": self.ctx.node, "principal": self.ctx.principal}

    def check_cancelled(self) -> None:
        if self.ctx.is_cancelled():
            raise TaskCancelled("任务已被取消或授权已撤销，停止执行")

    # ---------- Skill 调用 ----------
    def call(self, name: str, /, **args: Any) -> BaseModel:
        started = time.time()
        record = self._base_record("skill", name)
        attempts: list[dict] = []
        with span(f"skill:{name}", task_id=self.ctx.task_id, node=self.ctx.node, principal=self.ctx.principal) as ids:
            record.update(ids)
            try:
                self.check_cancelled()
                skill = self._admit(name)
                inp = skill.spec.input_model(**args)
                self._pre_tool_use(skill.spec, inp)
                key = self._idempotency_key(skill.spec, inp)
                record.update(version=skill.spec.version, args_hash=key[:16])
                self._approval_gate(skill.spec, key[:16])
                if key in self._cache:
                    record["status"] = "cache_hit"
                    return self._cache[key]
                # 单飞：同一任务里并行专家用相同参数调用同一技能时，只让一个真正执行，其余等待复用结果
                while True:
                    ev = self._flight.get(key)
                    if ev is None:
                        ev = threading.Event()
                        if self._flight.setdefault(key, ev) is ev:
                            break
                        continue
                    ev.wait(timeout=float(skill.spec.timeout_s) + 5)
                    if key in self._cache:
                        record["status"] = "cache_hit"
                        return self._cache[key]
                try:
                    out = self._run_with_retry(skill, inp, attempts)  # 预算检查与计费在每次真实调用时进行
                    if not isinstance(out, skill.spec.output_model):
                        raise HarnessDenied(f"{name} 输出不符合契约")
                    self._cache[key] = out
                    record["status"] = "ok"
                    return out
                finally:
                    self._flight.pop(key, None)
                    ev.set()
            except (HarnessDenied, ApprovalRequired, BudgetExceeded, SkillFailed, TaskCancelled) as e:
                record.update(status=type(e).__name__, error=str(e))
                raise
            except ValidationError as e:
                # pydantic 的错误文本含 input_value 原文：个人查询会随审计落盘，只记录类型与位置
                record.update(status="invalid_input",
                              error="; ".join(f"{err['loc']}: {err['msg']}" for err in e.errors(include_input=False))[:300])
                raise HarnessDenied(f"{name} 参数不符合契约: {e.errors()[0]['msg']}") from e
            except GraphInterrupt:
                record["status"] = "awaiting_approval"
                raise
            finally:
                record["duration_ms"] = int((time.time() - started) * 1000)
                if attempts and (len(attempts) > 1 or attempts[0].get("status") != "ok"):
                    record["attempts"] = attempts    # 重试的每一次真实外呼都要能对账，首败原因不能只活在最终错误里
                self.audit(record)

    def _admit(self, name: str):
        skill = self.registry.get(name)
        if skill is None:
            raise HarnessDenied(f"未注册的 Skill: {name}")
        if not self.policy.is_allowed(self.ctx.route, name):
            raise HarnessDenied(f"流程 {self.ctx.route} 无权调用 {name}")
        if self.policy.approval_level(name) == "deny":
            raise HarnessDenied(f"{name} 被审批门控禁止")
        self.budget.check()
        return skill

    def _approval_gate(self, spec, args_hash: str) -> None:
        """confirm 级调用暂停等人确认。审批绑定 Skill、版本与参数，参数或版本变化即失效。"""
        if self.policy.approval_level(spec.name) != "confirm":
            return
        ticket = {
            "type": "approval", "skill": spec.name, "version": spec.version, "args_hash": args_hash,
            "ttl_s": int(self.policy.approvals.get("ttl_s", 600)),
            "message": f"调用 {spec.name}（v{spec.version}）需要你确认",
        }
        try:
            from langgraph.types import interrupt
            decision = interrupt(ticket)
        except RuntimeError as e:  # 不在图内运行（如 MCP）：无法等待人，直接拒绝
            raise ApprovalRequired(spec.name) from e
        bound = all(decision.get(k) == ticket[k] for k in ("skill", "version", "args_hash"))
        if not (decision.get("approve") and bound):
            raise HarnessDenied(f"{spec.name} 审批未通过或已失效：{decision.get('reason', '版本/参数与审批时不一致')}")

    def _pre_tool_use(self, spec, inp: BaseModel) -> None:
        try:
            as_of = getattr(inp, "as_of", None)
            if as_of is not None and as_of > self.ctx.as_of:
                raise HarnessDenied(f"{spec.name} 请求的时点 {as_of} 晚于研究时点 {self.ctx.as_of}")
            for field_name in spec.egress_fields:
                value = getattr(inp, field_name)
                if isinstance(value, str):
                    self.egress.check_outbound(value)
        except HarnessDenied:
            raise
        except Exception as e:  # 安全检查自身出错：默认阻断
            raise HarnessDenied(f"前置检查不可用，默认阻断: {e}") from e

    def _run_with_retry(self, skill, inp: BaseModel, attempts: list[dict]) -> BaseModel:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            self.budget.check()                                # 总时限与预算种类无关：每次尝试前都复查
            if skill.spec.budget_kind:
                self.budget.check(skill.spec.budget_kind)      # 每次真实调用前重新检查
                self._charge(skill.spec.budget_kind)           # 每次真实调用都计费：重试不是免费的
            t0 = time.time()
            try:
                out = _call_with_timeout(lambda: skill.run(inp), skill.spec.timeout_s)
            except _SkillTimeout:
                last = SkillFailed(f"{skill.spec.name} 超时（{skill.spec.timeout_s}s）", retryable=True)
                attempts.append({"n": attempt + 1, "status": "timeout", "error": str(last),
                                 "duration_ms": int((time.time() - t0) * 1000)})
            except SkillFailed as e:
                last = e
                attempts.append({"n": attempt + 1, "status": "failed", "error": str(e)[:200],
                                 "duration_ms": int((time.time() - t0) * 1000)})
            except Exception as e:
                retry = isinstance(e, skill.spec.retryable)
                last = SkillFailed(f"{skill.spec.name} 失败: {e}", retryable=retry)
                attempts.append({"n": attempt + 1, "status": "failed", "error": str(e)[:200],
                                 "duration_ms": int((time.time() - t0) * 1000)})
            else:
                attempts.append({"n": attempt + 1, "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
                self.budget.check()                            # 返回后复查总时限：超限的结论不再提交
                return out
            if not getattr(last, "retryable", False) or attempt == MAX_RETRIES:
                break
            self.check_cancelled()
            time.sleep(min(2 ** attempt, 8))
        raise last  # type: ignore[misc]

    @staticmethod
    def _idempotency_key(spec, inp: BaseModel) -> str:
        payload = json.dumps(
            {"skill": spec.name, "v": spec.version, "in": inp.model_dump(mode="json")},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ---------- 模型调用 ----------
    def llm(self, role: str, system: str, user: str, schema: type[BaseModel], purpose: str) -> BaseModel:
        """模型调用同样受撤销、预算、脱敏、超时与审计约束。"""
        started = time.time()
        record = {**self._base_record("llm", purpose), "role": role}
        timeout_s, wall_left = LLM_TIMEOUT_CAP_S, self.budget.wall_left()
        with span(f"llm:{purpose}", task_id=self.ctx.task_id, node=self.ctx.node, role=role) as ids:
            record.update(ids)
            try:
                self.check_cancelled()
                self.budget.check("tokens", 1)
                remaining = self.budget.remaining("tokens")
                if remaining < MIN_LLM_TOKENS:      # 不够产出最小合法 JSON：直接失败，不发必败请求
                    raise BudgetExceeded(f"token 预算不足（剩余 {remaining:.0f}，至少需要 {MIN_LLM_TOKENS}）")
                cap = int(min(remaining, 8192))
                if "wall_clock_s" in self.budget.limits:      # 模型调用同样受总时限约束
                    timeout_s = max(1.0, min(LLM_TIMEOUT_CAP_S, wall_left))
                obj, usage = _call_with_timeout(
                    lambda: self.llm_client.json(role=role, system=system, user=self.egress.redact(user),
                                                 schema=schema, max_tokens=cap),
                    timeout_s,
                )
                self._charge("tokens", usage)
                record.update(tokens=usage)
                self.budget.check()                 # 调用返回后复查总时限：超时产生的结论不再提交
                record.update(status="ok")
                return obj
            except _SkillTimeout as e:
                record.update(status="BudgetExceeded", error=f"模型调用超时（{timeout_s:.0f}s）")
                raise BudgetExceeded(f"模型调用超时（{timeout_s:.0f}s，总时限剩余 {wall_left:.0f}s）") from e
            except Exception as e:
                # 模型侧可能已经花掉 token（修复重试失败/API 中断）：按真实消耗计费并写审计
                spent = int(getattr(e, "tokens", 0) or getattr(e, "advisor_tokens", 0) or 0)
                if spent:
                    self._charge("tokens", spent)
                    record["tokens"] = spent
                record.update(status=type(e).__name__, error=str(e)[:300])
                raise
            finally:
                record["duration_ms"] = int((time.time() - started) * 1000)
                self.audit(record)

    def decide(self, state: str, questions: dict[str, tuple[str, dict[str, str]]], purpose: str) -> dict[str, dict[str, float]]:
        """决策小模型调用：同样受撤销、脱敏与审计约束；本地模型不计 token 预算，但总时限仍生效。"""
        if self.decision_client is None:
            raise RuntimeError("未配置决策小模型")
        started = time.time()
        record = {**self._base_record("decision", purpose), "role": "decision"}
        with span(f"decision:{purpose}", task_id=self.ctx.task_id, node=self.ctx.node) as ids:
            record.update(ids)
            try:
                self.check_cancelled()
                self.budget.check()                 # 已超总时限的任务不再发起外部请求
                answers, meta = self.decision_client.choose(self.egress.redact(state), questions)
                record.update(status="ok", tokens=0, model=meta.get("model"))
                return answers
            except Exception as e:
                record.update(status=type(e).__name__, error=str(e)[:300])
                raise
            finally:
                record["duration_ms"] = int((time.time() - started) * 1000)
                self.audit(record)
