"""R100a/R102a 回归：代码前后边界为 Unicode 数字；敏感问题被空白/零宽/连字符拆开的代码
用「相邻数字对桥接」兜底找回，且兜底不把金额/日期拼成代码、不参与模型/名称的优先级。"""
import threading
import time

from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.nodes import _codes_of, _codes_of_loose
from advisor.graph.runner import ResearchService
from tests.fakes import CODE, NAME, FakeLLM, FakeSource, FakeWebTools, as_of

NBSP = " "
ZWSP = "\u200b"


def _svc(repo, policy, **kw):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    return ResearchService(deps, InMemorySaver(), **kw)


def _run(repo, policy, question: str) -> dict:
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick"), background=False)
    return repo.get_task(tid)


def test_non_ascii_digit_adjacency_is_not_a_code():
    """前后边界是 Unicode 数字：阿拉伯-印度数字紧邻的 6 位 ASCII 片段不能被截成代码。"""
    assert _codes_of("看看٥600519的财报") == set()
    assert _codes_of("看看600519٥的财报") == set()
    assert _codes_of("看看 600519 的财报") == {"600519"}


def test_non_ascii_digit_adjacent_not_researched(repo, policy):
    """端到端：'٥600519' 不识别为标的。"""
    task = _run(repo, policy, "看看٥600519的财报")
    assert task["status"] == "failed"
    assert "未能识别" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_loose_parse_bridges_pairs_but_rejects_money_and_dates():
    """宽松兜底只在『相邻数字对拼接恰为代码』时取候选：金额（元/万/亿/%）与日期对不得拼接；
    零宽/连字符/空白拆开的真代码要能找回；多个候选不猜。"""
    assert _codes_of_loose("我600" + NBSP + "519股成本价30元怎么办？") == {"600519"}
    assert _codes_of_loose("我600-519股成本价30元怎么办？") == {"600519"}
    assert _codes_of_loose("我600" + ZWSP + "519股成本价30元怎么办？") == {"600519"}
    assert _codes_of_loose("我的房贷余额600" + NBSP + "519元怎么办？") == set()          # 余额不是标的
    assert _codes_of_loose("我持仓\n2024-12-31\n2025-01-02的流水怎么看？") == set()     # 日期对不拼
    assert _codes_of_loose("我600" + NBSP + "519股和300" + NBSP + "750股都有") == {"600519", "300750"}
    assert _codes_of_loose("我持仓\n2024-12-31\n600\n519股成本价30元") == {"600519"}      # 不整串吞码


def _consent_task(repo, policy, question: str) -> dict:
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick"), background=False)
    task = repo.get_task(tid)
    state = svc.graph.get_state({"configurable": {"thread_id": tid}}).values
    return {**task, "_state_ticker": state.get("ticker")}


def test_sensitive_split_code_recovered_and_state_consistent(repo, policy):
    """被 NBSP/换行拆开的真代码：宽松兜底找回，request.ticker 与图内 state.ticker 一致，
    敏感问题在 authorize 必然停在 waiting_approval。"""
    for q in ("我600" + NBSP + "519股成本价30元怎么办？", "我600\n519股成本价30元怎么办？"):
        task = _consent_task(repo, policy, q)
        assert task["status"] == "waiting_approval", task.get("error")
        assert task["request"]["ticker"] == "600519"
        assert task["_state_ticker"] == "600519"
        assert "600519" in task["interrupt"]["message"]           # 审批文案展示推断出的标的


def test_sensitive_amount_is_not_bound_as_code(repo, policy):
    """R102a important：敏感问题里的房贷/存款金额不得被拼成真实股票代码（600519/600000）。"""
    task = _run(repo, policy, "我的房贷余额600" + NBSP + "519元怎么办？")
    assert task["status"] == "failed"
    assert "未能识别" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_sensitive_date_pair_not_bound(repo, policy):
    task = _run(repo, policy, "我持仓\n2024-12-31\n2025-01-02的流水怎么看？")
    assert task["status"] == "failed"
    assert task["request"].get("ticker") is None


def test_loose_ambiguous_codes_not_guessed(repo, policy):
    """两个拆分候选（都带『股』）不硬猜：按未识别失败，不绑定其中之一。"""
    task = _run(repo, policy, "我600" + NBSP + "519股和300" + NBSP + "750股都有，怎么办？")
    assert task["status"] == "failed"
    assert task["request"].get("ticker") is None


def test_queued_cancel_backfills_nbsp_split_code(repo, policy):
    """排队中被取消（intake 未执行）：runner 落盘也要用同一宽松兜底回填拆开的代码，重试可用。"""
    svc = _svc(repo, policy, max_workers=1)
    gate = threading.Event()
    svc._pool.submit(lambda: gate.wait(10))
    try:
        tid = svc.start(ResearchRequest(question="我600" + NBSP + "519股成本价30元怎么办？",
                                        as_of=as_of(), route="quick"), background=True)
        task = repo.get_task(tid)
        assert task["status"] == "queued"
        assert task["request"]["ticker"] == "600519"
        svc.deps.repo.request_cancel(tid)
    finally:
        gate.set()
    deadline = time.time() + 10
    while time.time() < deadline and repo.get_task(tid)["status"] not in ("cancelled", "done", "failed"):
        time.sleep(0.02)
    assert repo.get_task(tid)["status"] == "cancelled"


def test_explicit_ticker_conflict_is_visible(repo, policy):
    """页面/请求指定的标的与问题点名的公司不一致：intake 事件必须说明（small 分类器路径也可见）。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=f"{NAME} 最近怎么样？", as_of=as_of(), route="quick",
                                    ticker="000001"), background=False)
    task = repo.get_task(tid)
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("指定标的是 000001" in m and NAME in m for m in notes)
