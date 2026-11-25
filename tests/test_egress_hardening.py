"""出网脱敏回归：零宽/全角/分隔符归一、+86 与邮箱令牌、持仓窗口、命中后全文遮蔽、落盘不存原文。"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.harness import HarnessDenied
from advisor.harness.egress import EgressGuard
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of


@pytest.fixture
def guard(policy):
    return EgressGuard(policy.egress.get("sensitive_patterns", []), policy.egress.get("trade_intent_patterns", []))


def test_zero_width_chars_do_not_bypass(guard):
    text = "本​人​账​户 需要保护，不要外传"
    assert guard.find_sensitive(text)                       # 零宽拆开关键词不能再绕过正则
    with pytest.raises(HarnessDenied):
        guard.check_outbound(text)


def test_holding_with_stock_code_and_gap_detected(guard):
    for text in ("我有600519 1000股，要不要卖", "本人有 600036 200股", "家里持有 000001 500股",
                 "我有600519这个票1000股"):                   # 最后一条中间隔 9 字：只靠放宽窗口才命中，归一合并救不了
        assert guard.find_sensitive(text), text             # 代码+空白超过旧 6 字窗口也不能漏检


def test_phone_account_email_token_variants_detected(guard):
    for text in ("+86 138-1234-5678", "138 1234 5678", "１３８１２３４５６７８",
                 "６２２２ ０２０２ ００００ １２３４", "联系 someone@example.com",
                 "key=sk-abcdefghijklmnopqrstuvwx123456"):
        assert guard.find_sensitive(text), text             # +86/分隔/全角/邮箱/令牌都有模式


def test_redact_masks_all_numbers_after_sensitive_hit(guard):
    text = "我的持仓有 30万元，成本价 6.2 元，账户 6222 0202 0000 1234\n[ev_abc123 | A级 | calc] 营收 910 亿元"
    out = guard.redact(text)
    first, second = out.split("\n")
    assert "[已脱敏]" in first
    assert not any(ch.isdigit() for ch in first)            # consent 承诺「已脱敏」：成本/账号不能原样出域
    assert "ev_abc123" in second and "910" in second        # 其他行（提示词证据区）不受影响


def test_redact_leaves_public_text_untouched(guard):
    text = "复核 600999 半年报 2026-09-30 的业绩与估值"
    assert guard.redact(text) == text                       # 无个人敏感信息：原文不变


def test_personal_question_not_persisted_raw(repo, policy):
    """consent 承诺「问题中的个人信息已脱敏」：任务请求与档案里都不能留原文金额/代码。"""
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    svc = ResearchService(deps, InMemorySaver())
    tid = svc.start(ResearchRequest(question=f"我的持仓有 30万元 {CODE}，要不要卖？", as_of=as_of(), route="standard"),
                    background=False)
    task = repo.get_task(tid)
    assert task["status"] == "waiting_approval" and task["interrupt"]["type"] == "consent"
    svc.resume(tid, approve=True, background=False)
    task = repo.get_task(tid)
    assert task["status"] == "done", task
    assert "[已脱敏]" in task["request"]["question"] and "30" not in task["request"]["question"]
    d = repo.get_dossier(CODE)
    assert d is not None and "[已脱敏]" in d.question
    assert "30万" not in d.question and CODE not in d.question   # 原文数字不进档案；代码单独存 ticker 字段
    assert d.ticker == CODE
