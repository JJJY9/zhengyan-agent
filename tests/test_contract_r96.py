"""R96a 回归：落盘回填 request.ticker 的安全边界（多代码/无代码不误写、显式不覆盖、
空白归一、名称回填）与 watch 复核 429 的真实领取→释放链路。"""
import threading
import time
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver

from advisor.api.watch import review
from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.services import Services
from advisor.watch.models import Alert
from tests.fakes import CODE, NAME, FakeLLM, FakeSource, FakeWebTools, as_of
from tests.test_phase2 import make_service


def _svc(repo, policy, **kw):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    return ResearchService(deps, InMemorySaver(), **kw)


def _run_foreground(repo, policy, question: str, ticker: str | None = None) -> dict:
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick", ticker=ticker))
    return repo.get_task(tid)["request"]


def test_backfill_skips_multiple_or_missing_codes(repo, policy):
    """回填只在「恰好一个 A 股代码」时发生：多代码不猜、财务数字（208448 万元）不是代码。"""
    req = _run_foreground(repo, policy, f"复核 {CODE} 和 300750 哪个好？")
    assert req.get("ticker") is None                       # 多代码：不硬猜其中一个
    req = _run_foreground(repo, policy, "公司营收 208448 万元说明什么？")
    assert req.get("ticker") is None                       # 非 A 股形态的 6 位数字不得当成代码持久化


def test_backfill_keeps_explicit_ticker(repo, policy):
    """客户端显式给了 ticker：即使问题里还有别的代码，也不得被回填覆盖。"""
    req = _run_foreground(repo, policy, f"复核 {CODE} 半年报业绩", ticker="000001")
    assert req.get("ticker") == "000001"


def test_backfill_normalizes_whitespace_and_fullwidth_codes(repo, policy):
    """代码内空白（半角/全角空格）与全角数字要被归一成合法 ticker：写入值必须匹配 ^[0-9]{6}$，
    否则重试按原参数发起会在请求校验处 422。"""
    req = _run_foreground(repo, policy, "看看６００　５１９的财报")   # 全角数字 + 全角空格分隔
    assert req.get("ticker") == "600519"
    req = _run_foreground(repo, policy, "看看 300 750 的走势")        # 半角空格分隔
    assert req.get("ticker") == "300750"


def test_name_only_question_retries_via_intake(repo, policy):
    """名称型问题不把档案名写成显式 ticker（R98a：那会越过 intake 的模型解析、可能研究错公司）；
    排队取消后用保存的请求重试，由 intake 自身的名称兜底落到正确档案。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})   # as_of 晚于任务：不做 prior 装载
    svc = _svc(repo, policy, max_workers=1)
    gate = threading.Event()
    svc._pool.submit(lambda: gate.wait(10))                # 占住唯一 worker：任务保持排队
    try:
        tid = svc.start(ResearchRequest(question=f"{NAME} 最近的业绩怎么样？", as_of=as_of(), route="quick"),
                        background=True)
        task = repo.get_task(tid)
        assert task["status"] == "queued"
        assert task["request"].get("ticker") is None       # 名称启发式不写成权威值
        svc.deps.repo.request_cancel(tid)
    finally:
        gate.set()
    deadline = time.time() + 10
    while time.time() < deadline and repo.get_task(tid)["status"] not in ("cancelled", "done", "failed"):
        time.sleep(0.02)
    assert repo.get_task(tid)["status"] == "cancelled"
    # 按 TaskDetail 的方式用保存的请求重试：intake 的名称解析兜底仍能识别标的
    req = repo.get_task(tid)["request"]
    tid2 = svc.start(ResearchRequest(question=req["question"], as_of=req["as_of"], route=req["route"]),
                     background=False)
    task2 = repo.get_task(tid2)
    assert task2["status"] == "done", task2.get("error")
    assert task2["result"]["dossier_key"] == CODE            # intake 名称解析落到了正确标的


def test_watch_review_pool_saturated_releases_alert(repo, policy):
    """队列满时复核返回 429 且领取已释放（提醒仍 open、无半提交任务）；名额恢复后可重试成功。"""
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    services.watch.store.subscribe(CODE, NAME, 10)
    services.watch.store.add_alert(Alert(id="al_r96", ticker=CODE, name=NAME, dossier_key=CODE,
                                         dossier_version=1, fingerprint="fp_r96", events=[], affected=[],
                                         created_at=datetime.now(timezone.utc)))
    svc.max_pending = 0
    with pytest.raises(HTTPException) as ei:
        review("al_r96", services)
    assert ei.value.status_code == 429 and "队列已满" in ei.value.detail
    assert services.watch.store.get_alert("al_r96").status == "open"    # claim_review 已被释放：可重试
    assert not repo.list_tasks()                                        # start 失败没有留下任务文档
    svc.max_pending = 64
    out = review("al_r96", services)                                    # 名额恢复后重试成功
    alert = services.watch.store.get_alert("al_r96")
    assert alert.status == "reviewing" and alert.review_task_id == out["task_id"]
