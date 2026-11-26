"""深研专家审批重放回归：搜索词生成确定性复用——恢复后已批准的检索必须按原参数执行，
不能被重采样后的参数判成「审批未通过」。"""
import time

from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.experts.schemas import ExpertQueries
from advisor.experts.specs import EXPERTS
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of


class _ResamplingLLM(FakeLLM):
    """每次被调用都生成不同的搜索词：模拟真实 LLM 在节点重放时不保证确定性。"""

    def __init__(self):
        super().__init__()
        self.expert_q_calls = 0

    def json(self, role, system, user, schema, max_tokens=None):
        out = super().json(role, system, user, schema, max_tokens=max_tokens)
        if schema is ExpertQueries:
            self.expert_q_calls += 1
            obj, usage = out
            obj.queries = [f"示例酒业 业绩 {self.expert_q_calls}"]
            return obj, usage
        return out


def test_approved_searches_replay_with_same_queries(repo, policy):
    policy.approvals.setdefault("skills", {})["web.search"] = "confirm"
    web, llm = FakeWebTools(), _ResamplingLLM()
    deps = build_deps(repo, llm, policy, source=FakeSource(), web_tools=web)
    svc = ResearchService(deps, InMemorySaver())
    tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩，多空分歧在哪里", as_of=as_of(), route="deep"),
                    background=False)
    rounds = 0
    while rounds < 40:
        task = deps.repo.get_task(tid)
        if task["status"] != "waiting_approval":
            break
        time.sleep(0.05)
        svc.resume(tid, approve=True, background=False)
        rounds += 1
    task = deps.repo.get_task(tid)
    assert task["status"] == "done", task
    searching = sum(1 for key in ("fundamental", "valuation", "technical", "red_team") if EXPERTS[key].searches)
    assert llm.expert_q_calls == searching                    # 每个专家只生成一次搜索词（重放不重采样）
    denied = [a for a in deps.repo.list_audit(tid)
              if a.get("name") == "web.search" and a.get("status") == "HarnessDenied"]
    assert denied == []                                       # 用户同意的审批不能因重采样对不上参数而被记成未通过
    assert len(web.searches) == searching + 1                 # 每人一次真实搜索；反证官另固定前置一条「风险 质疑」
