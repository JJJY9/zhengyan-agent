"""决策入口与路由判别的回归：非有限概率、温度与阈值边界、代码解析、敏感/交易规则、质量门升级。"""
import json

from advisor.contracts import Claim
from advisor.contracts.deliberation import RouteDecision
from advisor.graph.nodes import Nodes
from advisor.harness import Budget
from advisor.harness.egress import EgressGuard
from advisor.routing import calibrate, route_from_probs
from advisor.routing.router import normalize
from advisor.skills.research.earnings_review import IntakeOut
from tests.fakes import FakeLLM
from tests.test_phase2 import make_service, run

CFG = {"temperature": 0.75, "abstain_below": 0.55, "deep_threshold_stakes": "high"}


# ---------- 路由数学 ----------

def test_non_finite_probabilities_are_sanitized():
    probs = {"quick": float("nan"), "standard": 0.3, "deep": 0.4}
    norm = normalize(probs)
    assert norm["quick"] == 0.0 and abs(sum(norm.values()) - 1.0) < 1e-9
    d = route_from_probs(probs, CFG)
    assert d.route == "deep" and not d.abstain            # NaN 不能被静默当作快查
    json.dumps(d.model_dump(mode="json"))                 # 决策必须可序列化（NaN 落库会让接口 500）
    inf = route_from_probs({"quick": float("inf"), "standard": 0.3, "deep": 0.4}, CFG)
    assert inf.route == "deep"


def test_invalid_temperature_is_neutral():
    dist = {"quick": 0.8, "standard": 0.1, "deep": 0.1}
    flat = calibrate(dist, 1.0)
    assert calibrate(dist, 0) == flat and calibrate(dist, -1) == flat   # 0/负数不得除零或反转排序
    d = route_from_probs(dist, {"temperature": 0, "abstain_below": 0.55})
    assert d.route == "quick" and not d.abstain


def test_force_deep_uses_stakes_threshold_ordering():
    probs = {"quick": 0.7, "standard": 0.2, "deep": 0.1}
    cfg = {"temperature": 0.75, "abstain_below": 0.55, "deep_threshold_stakes": "mid"}
    assert route_from_probs(probs, cfg, stakes="high", uncertainty="high").route == "deep"
    assert route_from_probs(probs, cfg, stakes="mid", uncertainty="high").route == "deep"
    assert route_from_probs(probs, cfg, stakes="low", uncertainty="high").route == "quick"


def test_abstain_uses_unrounded_probability():
    # 校准后真实最高概率 0.5499943（四舍五入显示才是 0.55）：按「低于阈值」应拒判
    probs = {"quick": 0.4943, "standard": 0.25285, "deep": 0.25285}
    d = route_from_probs(probs, CFG, stakes="low", uncertainty="low")
    assert d.abstain and d.route == "standard"


# ---------- intake：代码解析与敏感信息 ----------

class _TickerLLM(FakeLLM):
    def __init__(self, ticker, **kw):
        super().__init__(**kw)
        self.ticker = ticker

    def json(self, role, system, user, schema, max_tokens=None):
        if schema is IntakeOut:
            self.calls.append((role, "IntakeOut"))
            return IntakeOut(company_name="五粮液", ticker=self.ticker), 50
        return super().json(role, system, user, schema, max_tokens)


def test_intake_prefers_model_ticker_over_amount_digits(repo, policy):
    svc = make_service(repo, policy, _TickerLLM("000858"))
    tid, task = run(svc, question="五粮液2023年研发费用300750万元，证券代码000858，毛利率是多少？")
    assert task["status"] == "done", task
    assert task["result"]["dossier_key"] == "000858"      # 金额 300750 不再被当成股票代码


def test_intake_rejects_ambiguous_multiple_codes(repo, policy):
    svc = make_service(repo, policy)
    tid, task = run(svc, question="比较600519和000858哪个好")
    assert task["status"] == "failed" and "多个股票代码" in task["error"]


def test_intake_normalizes_fullwidth_code(repo, policy):
    svc = make_service(repo, policy)
    tid, task = run(svc, question="６００５１９最新业绩怎么样")
    assert task["status"] == "done", task
    assert task["result"]["dossier_key"] == "600519"


def test_sensitive_named_holding_reaches_consent(repo, policy):
    """名字型持仓问题：脱敏不能让公司名消失，入口应识别代码并进入同意流程（而不是直接 failed）。"""
    svc = make_service(repo, policy)
    run(svc, question="复核 600999 半年报业绩")                     # 先有「示例酒业」档案供名字解析
    tid, task = run(svc, question="我持有示例酒业1000股，成本价30元，该怎么办")
    assert task["status"] == "waiting_approval"
    assert task["interrupt"]["type"] == "consent"


def test_sensitive_patterns_first_person_only(policy):
    guard = EgressGuard(policy.egress["sensitive_patterns"], policy.egress["trade_intent_patterns"])
    assert guard.find_sensitive("我有贵州茅台1000股，成本价30元，该怎么办")
    assert guard.find_sensitive("我重仓600519")
    assert not guard.find_sensitive("北向资金持有600519 1000万股，这意味着什么")
    masked = guard.redact_amounts("我持有贵州茅台1000股，成本价30元")
    assert "贵州茅台" in masked and "1000" not in masked    # 入口识别保留公司名，只脱敏金额


def test_trade_intent_covers_bare_verbs(policy):
    guard = EgressGuard(policy.egress["sensitive_patterns"], policy.egress["trade_intent_patterns"])
    for q in ("帮我买100股 600999", "给我买600519", "卖600999", "Buy 600519 now"):
        assert guard.has_trade_intent(q), q
    assert not guard.has_trade_intent("要不要买600519")     # 疑问句仍按提问
    assert not guard.has_trade_intent("买入评级如何")       # 评级语境不是下单


def test_bare_buy_intent_adds_boundary_note(repo, policy):
    svc = make_service(repo, policy)
    tid, task = run(svc, question="帮我买100股 600999，现在合适吗")
    assert task["status"] == "done", task
    d = repo.get_dossier("600999")
    assert any("不执行任何买卖" in n for n in d.boundary_notes)


# ---------- 质量门：反方观点单独触发升级 ----------

def test_quality_gate_escalates_on_dissents_alone(repo, policy):
    svc = make_service(repo, policy)
    nodes = Nodes(svc.deps)
    claims = [Claim(id=f"c{i}", text=f"事实{i}", kind="fact", status="verified").model_dump(mode="json")
              for i in range(4)]
    state = {
        "task_id": "t_gate", "route": "standard", "iteration": 0,
        "budget": Budget(limits={"tokens": 1000, "search_calls": 5}).to_dict(),
        "claims": claims, "calc_ids": ["ev_x"], "evidence": {},
        "routing": RouteDecision(route="standard", source="model").model_dump(mode="json"),
        "draft": {"dissents": ["反方一", "反方二", "反方三"], "search_gaps": []},
    }
    upd = nodes.quality_gate(state)
    assert upd["quality"]["next"] == "escalate" and upd["route"] == "deep"
