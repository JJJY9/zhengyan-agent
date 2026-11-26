"""R79a 脱敏性质回归：用确定性随机组合（分隔符 × 号码类型 × 零宽/全角）批量验证三条不变量——
命中即遮蔽（命中行不残留任何数字）、幂等、公开文本不变；证据区相邻日期不误伤。"""
import random

from advisor.harness.egress import EgressGuard
from advisor.harness.policy import Policy

RNG = random.Random(20261127)

SEPS = ["", " ", "  ", "\n", "\n\n", "-", " – ", "\t", "​", "　", "\r\n"]


def _guard() -> EgressGuard:
    p = Policy.load()
    return EgressGuard(p.egress["sensitive_patterns"], p.egress.get("trade_intent_patterns", []))


def _seps(rng):
    return rng.choice(SEPS), rng.choice(SEPS), rng.choice(SEPS)


def _cases(rng, n=120):
    out = []
    for _ in range(n):
        a, b, c = _seps(rng)
        kind = rng.choice(["phone", "card", "id", "holding"])
        if kind == "phone":
            out.append(f"我的手机号 138{a}1234{b}5678{c}请联系我")
        elif kind == "card":
            out.append(f"银行卡{6}{a}222{b}0202{c}0000 1234 5678 9012 请核对该卡")
        elif kind == "id":
            out.append(f"身份证号 110101{a}19900307{b}123X 请登记{c}一下")
        else:
            out.append(f"我有贵州茅台{a}10{b}00 股，成本{c}1500 元，该怎么办")
    return out


def test_hit_implies_all_digits_masked():
    guard = _guard()
    for text in _cases(RNG):
        assert guard.find_sensitive(text), text
        red = guard.redact_question(text)
        assert not any(ch.isdigit() for ch in red), (text, red)


def test_redact_is_idempotent():
    guard = _guard()
    for text in _cases(RNG):
        once = guard.redact(text)
        assert guard.redact(once) == once, text


def test_public_text_unchanged():
    guard = _guard()
    words = ["复核", "600999", "半年报", "业绩", "估值", "营收", "910亿元", "2026-09-30", "同比", "+1.11%"]
    for _ in range(80):
        rng = random.Random(RNG.random())
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 8)))
        assert guard.redact(text) == text, text


def test_adjacent_dates_and_numbers_survive():
    guard = _guard()
    texts = [
        "证据：\n2024-12-31\n2025-03-31 营收 910亿元",
        "2023-06-30\n2023-09-30\n2023-12-31 净利润 12.5 亿元",
        "报告期 2026-09-30 收盘 1234.56 元",
        "报告期 2024-12-31 2025-03-31 营收 910亿元",
    ]
    for text in texts:
        assert guard.find_sensitive(text) == [], text
        assert guard.redact(text) == text, text


def test_punctuation_between_number_and_unit_detected():
    """数字与单位之间插入符号（「1000-股」）不能绕过识别：consent 必须触发、问题文本数字全遮蔽。"""
    guard = _guard()
    for punct in ("-", "–", "—", "~"):
        text = f"我有贵州茅台1000{punct}股，成本1500元，该怎么办"
        assert guard.find_sensitive(text), text
        red = guard.redact_question(text)
        assert "1000" not in red and "1500" not in red and "贵州茅台" in red


def test_task_request_question_fully_masked(repo, policy):
    """任务落盘的请求与档案问题用 redact_question：整段问题里的数字全部消失（含未被单条模式覆盖的成本）。"""
    from advisor.contracts import ResearchRequest
    from tests.fakes import CODE, FakeLLM, as_of
    from tests.test_phase2 import make_service

    svc = make_service(repo, policy, llm=FakeLLM())
    tid = svc.start(ResearchRequest(question=f"我有贵州茅台\n1000\n股，成本1500 元，想了解 {CODE} 半年报",
                                    as_of=as_of(), route="standard"), background=False)
    stored = svc.deps.repo.get_task(tid)["request"]["question"]
    assert not any(ch.isdigit() for ch in stored), stored
    svc.resume(tid, approve=True, background=False)
    d = svc.deps.repo.get_dossier(CODE)
    assert d is not None and not any(ch.isdigit() for ch in d.question), d.question
