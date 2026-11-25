"""审批恢复与预算账目回归：多条查询逐轮审批收敛、多专家同时待审批按中断 id 恢复、
节点被审批中断后计费不丢（route 上限仍守住）、审批等待不计入执行时限、重试逐次留痕。"""
import time
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from advisor.contracts import ResearchRequest
from advisor.contracts.skill import Skill, SkillSpec
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.harness import Budget, BudgetExceeded, Harness, RunContext, SkillFailed
from advisor.skills.registry import SkillRegistry
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    n: int = 1


def _confirm_service(repo, policy, llm=None):
    policy.approvals.setdefault("skills", {})["web.search"] = "confirm"
    web = FakeWebTools()
    deps = build_deps(repo, llm or FakeLLM(), policy, source=FakeSource(), web_tools=web)
    return ResearchService(deps, InMemorySaver()), web


def _approve_until_done(svc, tid, max_rounds=40):
    task = svc.deps.repo.get_task(tid)
    for _ in range(max_rounds):
        if task["status"] != "waiting_approval":
            break
        time.sleep(0.05)                                  # 模拟真人点击审批的停留：暂停时长必须不破坏恢复
        svc.resume(tid, approve=True, background=False)
        task = svc.deps.repo.get_task(tid)
    return task


# ---------- 多条查询的审批恢复 ----------

def test_multi_query_confirm_resumes_converge(repo, policy):
    """标准流程逐条审批：每轮同意都要投递给用户看到的那次调用，任务最终完成而不是原地打转。"""
    svc, web = _confirm_service(repo, policy)
    tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩与估值", as_of=as_of(), route="standard"),
                    background=False)
    task = _approve_until_done(svc, tid)
    assert task["status"] == "done", task
    denied = [a for a in repo.list_audit(tid) if a.get("name") == "web.search" and a.get("status") == "HarnessDenied"]
    assert denied == []                                   # 用户每次都点了同意，不能被记成「审批未通过」
    for q in ("示例酒业 业绩", "示例酒业 行业库存", "示例酒业 风险"):
        assert any(q in s for s in web.searches)          # 三条查询都真的执行了


def test_parallel_expert_interrupts_resume_by_id(repo, policy):
    """深研多个专家同时停在审批：票据带 interrupt id 并定向恢复，不能整单 failed 或无限等待。"""
    svc, web = _confirm_service(repo, policy)
    tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩，多空分歧在哪里", as_of=as_of(), route="deep"),
                    background=False)
    task = repo.get_task(tid)
    assert task["status"] == "waiting_approval"
    assert task["interrupt"].get("interrupt_id")          # 多中断并存时裸值会被错误任务抢走，必须按 id 恢复
    task = _approve_until_done(svc, tid, max_rounds=60)
    assert task["status"] == "done", task
    assert web.searches                                   # 至少一条查询真实执行后才可能完成


# ---------- 中断后的预算账目 ----------

def test_budget_charges_survive_interrupt(repo, policy):
    """中断前已消耗的检索不能丢账：route 的 search_calls 上限要守住，结果账目与实际外呼一致。"""
    policy.routes["standard"].budget["search_calls"] = 1
    svc, web = _confirm_service(repo, policy)
    tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩与估值", as_of=as_of(), route="standard"),
                    background=False)
    task = _approve_until_done(svc, tid)
    assert task["status"] == "done", task
    assert len(web.searches) == 1                         # 上限 1：恢复后不能凭丢账再打一次真实外呼
    assert task["result"]["budget"]["used"]["search_calls"] == 1


def test_paused_time_excluded_from_wall_clock():
    """总时限只统计执行时间：100s 里 80s 在等审批，剩余约 30s 而不是已超时。"""
    b = Budget.from_dict({"limits": {"wall_clock_s": 50, "tokens": 100}, "used": {},
                          "started_at": time.time() - 100, "paused_s": 80})
    b.check()                                             # 不抛 BudgetExceeded
    assert 28 <= b.wall_left() <= 32


# ---------- 模型调用受剩余时限约束 ----------

class _SlowClient:
    def __init__(self, delay: float):
        self.delay = delay

    def json(self, role, system, user, schema, max_tokens=None):
        time.sleep(self.delay)
        return schema(n=1), 5


def test_llm_timeout_uses_remaining_wall_clock(policy):
    """模型调用超时按剩余时限取，而不是每次都拿满 300s 上限。"""
    budget = Budget(limits={"wall_clock_s": 5, "tokens": 10000}, started_at=time.time() - 4.6)
    audit: list[dict] = []
    h = Harness(policy, SkillRegistry([]), audit.append, budget,
                RunContext("t", "standard", date(2026, 10, 18)), llm=_SlowClient(1.3))
    t0 = time.time()
    with pytest.raises(BudgetExceeded, match="超时"):
        h.llm("draft", "s", "u", _Out, purpose="slow")
    assert time.time() - t0 < 1.25                        # 约 1s 的剩余时限，而不是等慢客户端跑完


# ---------- 重试逐次留痕 ----------

def test_retry_attempts_recorded_in_audit(policy, monkeypatch):
    monkeypatch.setattr("advisor.harness.executor.time.sleep", lambda s: None)   # 跳过退避等待
    calls = {"n": 0}

    def run(inp):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SkillFailed("第一次瞬时失败", retryable=True)
        return _Out(n=2)

    sk = Skill(spec=SkillSpec(name="market.daily", version="1", category="data", permission="read", description="",
                              input_model=_In, output_model=_Out, timeout_s=2), run=run)
    audit: list[dict] = []
    h = Harness(policy, SkillRegistry([sk]), audit.append, Budget(limits={"wall_clock_s": 60}),
                RunContext("t", "standard", date(2026, 10, 18)))
    assert h.call("market.daily").n == 2
    assert [a["status"] for a in audit[-1]["attempts"]] == ["failed", "ok"]      # 首败原因不能只活在最终错误里
