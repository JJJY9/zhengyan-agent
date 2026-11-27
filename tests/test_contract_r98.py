"""R98a 回归：代码形态只认 ASCII 数字与空格分隔（制表符/换行/其他 Unicode 数字不拼码、
不落盘非法 ticker）；模型 ticker 与问题文本共用同一归一口径。"""
from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.nodes import _codes_of
from advisor.graph.runner import ResearchService
from advisor.skills.research.earnings_review import IntakeOut
from tests.fakes import FakeLLM, FakeSource, FakeWebTools, as_of


class _TickerLLM(FakeLLM):
    """intake 返回指定 ticker 的模型桩（其余角色沿用 FakeLLM）。"""

    def __init__(self, ticker: str):
        super().__init__()
        self._ticker = ticker

    def json(self, role, system, user, schema, max_tokens=None):
        if schema is IntakeOut:
            return IntakeOut(company_name="", ticker=self._ticker, intent="research", focus=""), 50
        return super().json(role, system, user, schema, max_tokens)


def _run(repo, policy, llm, question: str) -> dict:
    deps = build_deps(repo, llm, policy, source=FakeSource(), web_tools=FakeWebTools())
    svc = ResearchService(deps, InMemorySaver())
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick"), background=False)
    return repo.get_task(tid)


def test_codes_reject_tab_newline_and_unicode_digits():
    """制表符/换行分隔的两个无关数字不是代码（粘贴表格/折行不能被拼成合法 ticker）；
    阿拉伯-印度数字不是 A 股代码；半角/全角空格分隔仍可识别。"""
    assert _codes_of("营收600\n519家门店说明什么") == set()
    assert _codes_of("营收600\t519家门店说明什么") == set()
    assert _codes_of("看看600٥١٩的财报") == set()          # 阿拉伯-印度数字
    assert _codes_of("看看600 519的财报") == {"600519"}
    assert _codes_of("看看600　519的财报") == {"600519"}     # 全角空格


def test_model_ticker_whitespace_normalized_like_question(repo, policy):
    """模型 ticker 字段里的内部空白（制表符/全角空格）与问题文本同一归一：能识别的就接受。"""
    task = _run(repo, policy, _TickerLLM("600\t519"), "帮我看看这家公司的近况")
    assert task["status"] == "done", task.get("error")
    assert task["request"]["ticker"] == "600519"


def test_model_unicode_digits_not_persisted(repo, policy):
    """模型返回混合 Unicode 数字时不得落盘为 ticker（否则重试按原参数发起会 422）：
    按未识别处理，任务失败且 request.ticker 保持为空。"""
    task = _run(repo, policy, _TickerLLM("600٥١٩"), "帮我看看这家公司的近况")
    assert task["status"] == "failed"
    assert "未能识别" in (task.get("error") or "")
    assert task["request"].get("ticker") is None
