"""R100a 回归：代码前后边界恢复 Unicode 数字（非 ASCII 数字紧邻不算代码）、
敏感问题被 NBSP/换行拆开的代码用宽松合并兜底找回（否则脱敏后模型也看不到，任务必失败）。"""
from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.nodes import _codes_of, _codes_of_loose
from advisor.graph.runner import ResearchService
from tests.fakes import FakeLLM, FakeSource, FakeWebTools, as_of


def _run(repo, policy, question: str) -> dict:
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    svc = ResearchService(deps, InMemorySaver())
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick"), background=False)
    return repo.get_task(tid)


def test_non_ascii_digit_adjacency_is_not_a_code():
    """前后边界是 Unicode 数字：阿拉伯-印度数字紧邻的 6 位 ASCII 片段不能被截成代码。"""
    assert _codes_of("看看٥600519的财报") == set()
    assert _codes_of("看看600519٥的财报") == set()
    assert _codes_of("看看 600519 的财报") == {"600519"}


def test_non_ascii_digit_adjacent_not_researched(repo, policy):
    """端到端：'٥600519' 不识别为标的（修复前会截成 600519 并研究整只股票）。"""
    task = _run(repo, policy, "看看٥600519的财报")
    assert task["status"] == "failed"
    assert "未能识别" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_loose_parse_merges_all_whitespace():
    """宽松兜底与 egress 同口径：数字间的一切空白都合并（只在敏感且严格为空时使用）。"""
    assert _codes_of("营收600\n519家门店") == set()          # 严格：不拼码
    assert _codes_of_loose("营收600\n519家门店") == {"600519"}
    assert _codes_of_loose("营收600\xa0519家门店") == {"600519"}


def test_sensitive_question_nbsp_split_code_recovered(repo, policy):
    """含个人信息的问句里被 NBSP 拆开的代码：数字会被整体遮蔽、模型看不到原文，
    严格解析为空必须走宽松兜底，否则任务从「可完成」回归为「未能识别」。"""
    task = _run(repo, policy, "我600\xa0519股成本价30元怎么办？")
    assert task["status"] in ("waiting_approval", "done", "running"), task.get("error")
    assert task["request"]["ticker"] == "600519"
    assert "未能识别" not in (task.get("error") or "")


def test_sensitive_question_newline_split_code_recovered(repo, policy):
    task = _run(repo, policy, "我600\n519股成本价30元怎么办？")
    assert task["status"] in ("waiting_approval", "done", "running"), task.get("error")
    assert task["request"]["ticker"] == "600519"
