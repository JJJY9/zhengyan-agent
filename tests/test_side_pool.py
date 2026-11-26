"""侧栏作业池回归：RSI/评测等后台作业不再占用交互研究的线程与排队名额。"""
import threading

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.runner import PoolSaturated, ResearchService
from advisor.services import Services
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service


def _stuck(ev: threading.Event, out: list):
    ev.wait(timeout=5)
    out.append(1)


def test_side_jobs_saturation_does_not_block_research(repo, policy):
    svc = make_service(repo, policy)
    svc_side = ResearchService(svc.deps, InMemorySaver(), max_pending=1, side_workers=1)
    ev, out = threading.Event(), []
    svc_side.submit_side(_stuck, ev, out)          # 侧栏占用唯一名额（线程同时被占住）
    with pytest.raises(PoolSaturated):
        svc_side.submit_side(_stuck, ev, out)      # 侧栏自己的队列满：拒绝而不是拖垮研究
    try:
        tid = svc_side.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                             background=False)
        assert svc_side.deps.repo.get_task(tid)["status"] == "done"   # 研究名额与线程完全隔离
    finally:
        ev.set()


def test_services_submit_routes_to_side_pool(repo, policy):
    svc = make_service(repo, policy)
    calls: list = []
    svc.submit_side = lambda fn, *a: calls.append(fn) or "fut"        # 记录路由目标
    services = Services(svc.deps, svc)

    def job():
        return None

    assert services.submit(job) == "fut" and calls == [job]           # 页面作业不占研究池
