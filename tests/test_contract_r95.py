"""R95a 回归：任务落盘即回填股票代码（排队取消也能按原参数重试）、watch 复核队列满→429。"""
import threading
import time

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver

from advisor.api.watch import review
from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.runner import PoolSaturated, ResearchService
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of


def _svc(repo, policy, **kw):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    return ResearchService(deps, InMemorySaver(), **kw)


def test_start_backfills_ticker_before_intake(repo, policy):
    """排队中的任务被取消（intake 从未执行）也能凭落盘时回填的代码按原参数重试。"""
    svc = _svc(repo, policy, max_workers=1)
    gate = threading.Event()
    svc._pool.submit(lambda: gate.wait(10))                 # 占住唯一 worker：任务保持排队
    try:
        tid = svc.start(ResearchRequest(question=f"我的持仓有 30万元 {CODE}，要不要卖？", as_of=as_of(),
                                        route="standard"), background=True)
        task = repo.get_task(tid)
        assert task["status"] == "queued"
        assert task["request"]["ticker"] == CODE             # 落盘时已从原文解析回填（只存代码）
        assert "30万" not in task["request"]["question"]      # 原文仍脱敏
        svc.deps.repo.request_cancel(tid)
    finally:
        gate.set()
    deadline = time.time() + 10
    while time.time() < deadline and repo.get_task(tid)["status"] not in ("cancelled", "done", "failed"):
        time.sleep(0.02)
    task = repo.get_task(tid)
    assert task["status"] == "cancelled"
    assert task["request"]["ticker"] == CODE                 # 重试按钮可凭它发起


def test_watch_review_pool_saturated_is_429():
    """交互研究队列满时「用新数据重新研究」返回 429，而不是 500。"""

    def boom(*a, **k):
        raise PoolSaturated("后台任务队列已满（最多 64 个排队/运行），请稍后再试")

    svc = type("S", (), {})()
    svc.watch = type("W", (), {"store": type("St", (), {"get_alert": staticmethod(lambda _id: object())})(),
                               "start_review": staticmethod(boom)})()
    svc.research = object()
    with pytest.raises(HTTPException) as ei:
        review("al_x", svc)
    assert ei.value.status_code == 429 and "队列已满" in ei.value.detail
