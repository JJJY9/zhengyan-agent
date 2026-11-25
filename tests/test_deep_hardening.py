"""深研编排回归：审批中断、多条质询裁定、检索预算分摊、检索/回应失败可见、基本面专家保留。"""
import re
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.experts import agent
from advisor.experts.schemas import ChairOutput, ChallengeDraft, ChallengeSet, RulingDraft, TeamMember, TeamPlan
from advisor.graph.deep import DeepNodes
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.harness import Budget, SkillFailed
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of
from tests.test_phase2 import make_service


def _deep(svc, question=f"复核 {CODE} 半年报业绩，多空分歧在哪里"):
    return svc.start(ResearchRequest(question=question, as_of=as_of(), route="deep"), background=False)


# ---------- 审批中断必须冒泡 ----------

def test_deep_respects_approval_interrupt(repo, policy):
    policy.approvals.setdefault("skills", {})["web.search"] = "confirm"
    svc = make_service(repo, policy)
    tid = _deep(svc)
    task = repo.get_task(tid)
    assert task["status"] == "waiting_approval"
    assert task["interrupt"]["type"] == "approval"      # 深研专家分支不能吞掉审批控制流


# ---------- 同一主张多条质询：任一成立即存疑 ----------

class _DupChallengeLLM(FakeLLM):
    def json(self, role, system, user, schema, max_tokens=None):
        if schema is ChallengeSet:
            target = re.findall(r"\[((?!red_team)\w+-c\d+) \|", user)[0]
            return ChallengeSet(challenges=[
                ChallengeDraft(target_claim_id=target, kind="logic_gap", text="第一条质疑", evidence_ids=[], severity="mid"),
                ChallengeDraft(target_claim_id=target, kind="logic_gap", text="第二条质疑", evidence_ids=[], severity="high"),
            ]), 150
        if schema is ChairOutput:
            board = re.findall(r"\[(\w+-[cr]\d+) \|", user)
            chs = re.findall(r"\[(ch\d+)\] 反证官", user)
            return ChairOutput(
                summary="综合结论", adopted_claim_ids=board,
                rulings=[RulingDraft(challenge_id=chs[0], verdict="rejected", reason="第一条不成立", severity="mid"),
                         RulingDraft(challenge_id=chs[1], verdict="upheld", reason="第二条成立", severity="high")],
                synthesis_claims=[], dissents=[], unknowns=[], review_triggers=[], committee=[]), 400
        return super().json(role, system, user, schema, max_tokens)


def test_deep_any_upheld_challenge_marks_disputed(repo, policy):
    svc = make_service(repo, policy, _DupChallengeLLM())
    tid = _deep(svc)
    task = repo.get_task(tid)
    assert task["status"] == "done", task
    d = repo.get_dossier(CODE)
    assert d.status == "partial" and d.quality.decision == "partial"
    assert d.quality.disputed >= 1 and any(c.status == "disputed" for c in d.claims)
    assert {r["verdict"] for r in d.deliberation["rulings"]} == {"rejected", "upheld"}


# ---------- 检索预算不足时不能全员分 0 ----------

def test_dispatch_spreads_search_budget_below_team_size(repo, policy):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    nodes = DeepNodes(deps)
    team = TeamPlan(mode="debate", members=[
        TeamMember(expert="fundamental", questions=["业绩"]),
        TeamMember(expert="valuation", questions=["估值"]),
        TeamMember(expert="technical", questions=["走势"]),
    ])
    state = {"task_id": "t", "route": "deep", "as_of": as_of().isoformat(), "ticker": CODE, "company_name": "示例酒业",
             "question": "q", "evidence": {},
             "budget": Budget(limits={"tokens": 1_200_000, "search_calls": 3, "fetch_calls": 3,
                                      "wall_clock_s": 1800}).to_dict(),
             "team": team.model_dump(mode="json")}
    allocs = [s.arg["allocation"] for s in nodes.dispatch(state)]
    assert sum(a["search_calls"] for a in allocs) == 3 and sum(a["fetch_calls"] for a in allocs) == 3
    assert all(a["search_calls"] <= 2 for a in allocs)          # 单专家单轮上限仍生效
    assert len([a for a in allocs if a["search_calls"] > 0]) >= 1


# ---------- 检索失败进入档案未知项 ----------

class _DownWeb(FakeWebTools):
    def search(self, query, count):
        raise SkillFailed("搜索服务不可用")


def test_deep_search_failures_are_visible(repo, policy):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=_DownWeb())
    svc = ResearchService(deps, InMemorySaver())
    tid = _deep(svc)
    task = repo.get_task(tid)
    assert task["status"] == "done", task
    d = repo.get_dossier(CODE)
    assert any("搜索失败" in u for u in d.unknowns)             # 搜索全挂必须在档案未知项里可见


# ---------- 专家回应失败必须降级可见 ----------

class _NoRespondLLM(FakeLLM):
    def json(self, role, system, user, schema, max_tokens=None):
        if schema.__name__ == "ResponseSet":
            raise RuntimeError("回应服务不可用")
        return super().json(role, system, user, schema, max_tokens)


def test_deep_response_failure_degrades_quality(repo, policy):
    svc = make_service(repo, policy, _NoRespondLLM())
    tid = _deep(svc)
    task = repo.get_task(tid)
    assert task["status"] == "done", task
    d = repo.get_dossier(CODE)
    assert d.quality.decision == "partial"
    assert any("回应失败" in n for n in d.quality.notes)


# ---------- 截断后仍保留基本面专家 ----------

class _PlanLLM:
    def llm(self, role, system, user, schema, purpose=None):
        assert schema is TeamPlan
        return TeamPlan(mode="map_reduce", rationale="测试", members=[
            TeamMember(expert=e, questions=["q"]) for e in
            ("industry", "valuation", "technical", "event", "risk", "fundamental")])


def test_plan_team_keeps_fundamental_after_truncation(policy):
    plan = agent.plan_team(_PlanLLM(), SimpleNamespace(policy=policy), "q", "示例酒业（600999）", "high", "已有证据")
    names = [m.expert for m in plan.members]
    assert "fundamental" in names
    assert len(names) <= int(policy.experts.get("max_team", 5))
