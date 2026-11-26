"""R78a 回归：脱敏「检测=遮蔽」同口径的收敛项——名字保留、相邻日期不拼假号码、
生产提示词证据区不受问题区影响、名字型持仓在空仓库也能识别并进入 consent。"""
from advisor.contracts import ResearchRequest
from advisor.harness.egress import EgressGuard
from advisor.skills.research.earnings_review import IntakeOut
from tests.fakes import CODE, NAME, FakeLLM, as_of
from tests.test_phase2 import make_service


def _guard(policy):
    return EgressGuard(policy.egress.get("sensitive_patterns", []), policy.egress.get("trade_intent_patterns", []))


def test_company_name_kept_only_numbers_masked(policy):
    """含中文的个人描述保留文字（入口识别依赖公司名），只把数字整体遮蔽。"""
    guard = _guard(policy)
    text = "我有贵州茅台1000股，成本1500元，该怎么办"
    assert guard.find_sensitive(text)
    red = guard.redact(text)
    assert "贵州茅台" in red
    assert "1000" not in red and "1500" not in red
    assert guard.find_sensitive(red) == []


def test_adjacent_date_rows_not_merged(policy):
    """证据摘录里相邻日期行不能被拼成 16 位假卡号：原文保持，数字/日期可见。"""
    guard = _guard(policy)
    text = "证据：\n2024-12-31\n2025-03-31 营收 910亿元"
    assert guard.find_sensitive(text) == []
    assert guard.redact(text) == text


def test_production_prompt_keeps_evidence_numbers(repo, policy):
    """个人问题经预脱敏后进入提示词：证据区数字（910）保留，问题区金额不再出现。"""
    llm = FakeLLM()
    svc = make_service(repo, policy, llm=llm)
    tid = svc.start(ResearchRequest(question=f"我的持仓有 30万元，成本120元，想复核 {CODE} 半年报业绩与估值",
                                    as_of=as_of(), route="standard"), background=False)
    task = svc.deps.repo.get_task(tid)
    assert task["status"] == "waiting_approval" and task["interrupt"]["type"] == "consent"
    svc.resume(tid, approve=True, background=False)
    assert svc.deps.repo.get_task(tid)["status"] == "done"
    drafts = "\n".join(llm.prompts.get("DraftOutput", []))
    assert drafts and "910" in drafts                      # 证据区数字不被问题区的命中波及
    assert CODE in drafts                                   # 公司/代码上下文保留（模板中的代码不被桥接遮蔽）
    assert "30万" not in drafts                             # 问题区金额已脱敏
    assert "持仓有 30万元" not in drafts


class _NamedIntakeLLM(FakeLLM):
    """模拟真实模型：提示词里能看到公司名就给出对应代码（看不到则返回 null）。"""

    def json(self, role, system, user, schema, max_tokens=None):
        out = super().json(role, system, user, schema, max_tokens=max_tokens)
        if schema is IntakeOut:
            obj, usage = out
            obj.ticker = CODE if NAME in user else None
            return obj, usage
        return out


def test_named_holding_identified_in_empty_repo(repo, policy):
    """空仓库、未给 ticker：脱敏保留公司名，intake 仍能识别代码并进入 consent（R76 回归修复）。"""
    llm = _NamedIntakeLLM()
    svc = make_service(repo, policy, llm=llm)
    tid = svc.start(ResearchRequest(question=f"我持有{NAME}1000股，成本价30元，该怎么办",
                                    as_of=as_of(), route="standard"), background=False)
    task = svc.deps.repo.get_task(tid)
    assert task["status"] == "waiting_approval" and task["interrupt"]["type"] == "consent", task.get("error")
    prompt = "\n".join(llm.prompts.get("IntakeOut", []))
    assert NAME in prompt and "1000" not in prompt         # 公司名保留、数字已遮蔽
    svc.resume(tid, approve=True, background=False)
    assert svc.deps.repo.get_task(tid)["status"] == "done"
    assert svc.deps.repo.get_dossier(CODE) is not None
