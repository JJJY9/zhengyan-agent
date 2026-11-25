"""研究底稿核心回归：重复期选版、单季字段、比率边界、血缘、紧凑期号、pit 继承、核验口径、先取数后计划。"""
import re
from datetime import date, datetime, timezone
from decimal import Decimal

from advisor.contracts import Claim, Metric
from advisor.evidence import EvidenceFactory
from advisor.graph.deps import build_deps
from advisor.harness import Budget
from advisor.skills.data.structured import build_structured_skills
from advisor.skills.research.earnings_review import IntakeOut, QueryPlan, derive_metrics
from advisor.evidence.verifier import verify_claim
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools

AS_OF = date(2026, 10, 18)
B = 10 ** 9


def _metric(name, field, value, period, unit="元", basis="cumulative"):
    return Metric(name=name, field=field, value=Decimal(str(value)), unit=unit, period=period, basis=basis)


def _filing(metrics, published="2026-08-15", pit="strong"):
    f = EvidenceFactory(lambda h, c: None)
    return f.make("content" + published, level="A", source_type="filing", title="财报",
                  origin_id=f"filing:{CODE}:{metrics[0].period}", as_of=AS_OF,
                  published_at=datetime.fromisoformat(published).replace(tzinfo=timezone.utc),
                  excerpt="", metrics=metrics, skill="financials.statements", skill_version="1", pit=pit)


def _derive(repo, policy, statements):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    h = deps.harness("t_derive", "standard", AS_OF, Budget(limits={"tokens": 200000, "wall_clock_s": 60}))
    return derive_metrics(h, statements, deps.factory, AS_OF)


# ---------- 重复报告期选最新披露 ----------

class _DupSource(FakeSource):
    def income_rows(self, code):
        rows = super().income_rows(code)
        corrected = dict(rows[1])
        corrected.update({"OPERATE_INCOME": str(95 * B), "NOTICE_DATE": "2026-08-20 00:00:00",
                          "UPDATE_DATE": "2026-08-20 00:00:00"})
        return [corrected] + rows          # 更正版在前、原始版在后：不能按输入顺序取


def test_statements_keep_latest_correction_per_period():
    skills = {s.spec.name: s for s in build_structured_skills(_DupSource(), EvidenceFactory(lambda h, c: None))}
    out = skills["financials.statements"].run(skills["financials.statements"].spec.input_model(
        code=CODE, as_of=AS_OF, periods=8))
    periods = [c.metrics[0].period for c in out.cards]
    assert periods.count("2026-06-30") == 1
    card = next(c for c in out.cards if c.metrics[0].period == "2026-06-30")
    assert any(m.field == "OPERATE_INCOME" and m.value == Decimal(95 * B) for m in card.metrics)


def test_derive_prefers_latest_published_card(repo, policy):
    old = _filing([_metric("营业收入", "OPERATE_INCOME", 91 * B, "2026-06-30")], published="2026-08-15")
    new = _filing([_metric("营业收入", "OPERATE_INCOME", 95 * B, "2026-06-30")], published="2026-08-20")
    prev = _filing([_metric("营业收入", "OPERATE_INCOME", 54 * B, "2026-03-31")], published="2026-04-25")
    cards = _derive(repo, policy, [new, old, prev])   # 旧版排在后面，也不能覆盖新版
    assert cards and new.id in cards[0].derived_from and old.id not in cards[0].derived_from


# ---------- 单季字段与比率边界 ----------

def test_single_quarter_uses_own_field(repo, policy):
    cur = _filing([_metric("营业收入", "OPERATE_INCOME", 91 * B, "2026-06-30")])
    prev = _filing([_metric("营业收入", "OPERATE_INCOME", 54 * B, "2026-03-31")], published="2026-04-25")
    prior = _filing([_metric("营业收入", "OPERATE_INCOME", 90 * B, "2025-06-30")], published="2025-08-13")
    prior_q1 = _filing([_metric("营业收入", "OPERATE_INCOME", 50 * B, "2025-03-31")], published="2025-04-30")
    cards = _derive(repo, policy, [cur, prev, prior, prior_q1])
    fields = {m.field for m in cards[0].metrics}
    assert "OPERATE_INCOME_SQ" in fields and "OPERATE_INCOME_SQ_YOY" in fields
    assert "OPERATE_INCOME" not in fields                # 单季值不再占用累计字段


def test_ratio_skipped_without_current_and_for_negative_denominator(repo, policy):
    cur = _filing([_metric("营业收入", "OPERATE_INCOME", 91 * B, "2026-06-30"),
                   _metric("归母净利润", "PARENT_NETPROFIT", -5 * B, "2026-06-30"),
                   _metric("经营现金流净额", "NETCASH_OPERATE", -10 * B, "2026-06-30")])
    prev = _filing([_metric("营业收入", "OPERATE_INCOME", 54 * B, "2026-03-31")], published="2026-04-25")
    prior = _filing([_metric("营业收入", "OPERATE_INCOME", 90 * B, "2025-06-30"),
                     _metric("归母净利润", "PARENT_NETPROFIT", 45 * B, "2025-06-30"),
                     _metric("经营现金流净额", "NETCASH_OPERATE", 30 * B, "2025-06-30")], published="2025-08-13")
    cards = _derive(repo, policy, [cur, prev, prior])
    fields = {m.field for m in cards[0].metrics}
    assert "OCF_TO_NP" not in fields                     # 分母为负（双负）不算现金含量
    assert not any(m.period == "2025-06-30" and m.field == "OCF_TO_NP" for m in cards[0].metrics)


def test_point_growth_records_prior_lineage(repo, policy):
    cur = _filing([_metric("合同负债", "CONTRACT_LIAB", 31 * B, "2026-06-30", basis="point")])
    prior = _filing([_metric("合同负债", "CONTRACT_LIAB", 55 * B, "2025-06-30", basis="point")], published="2025-08-13")
    cards = _derive(repo, policy, [cur, prior])
    assert set(cards[0].derived_from) == {cur.id, prior.id}


def test_zero_base_growth_does_not_drop_single_quarter(repo, policy):
    cur = _filing([_metric("扣非归母净利润", "DEDUCT_PARENT_NETPROFIT", 4 * B, "2026-06-30")])
    prev = _filing([_metric("扣非归母净利润", "DEDUCT_PARENT_NETPROFIT", int(1.5 * B), "2026-03-31")],
                   published="2026-04-25")
    prior = _filing([_metric("扣非归母净利润", "DEDUCT_PARENT_NETPROFIT", 0, "2025-06-30")], published="2025-08-13")
    cards = _derive(repo, policy, [cur, prev, prior])
    fields = {m.field for m in cards[0].metrics}
    assert "DEDUCT_PARENT_NETPROFIT_SQ" in fields        # 基期 0 只影响同比


def test_compact_periods_supported(repo, policy):
    cur = _filing([_metric("营业收入", "OPERATE_INCOME", 91 * B, "20260630")])
    prev = _filing([_metric("营业收入", "OPERATE_INCOME", 54 * B, "20260331")], published="2026-04-25")
    prior = _filing([_metric("营业收入", "OPERATE_INCOME", 90 * B, "20250630")], published="2025-08-13")
    cards = _derive(repo, policy, [cur, prev, prior])
    assert any(m.basis == "single_quarter" for m in cards[0].metrics)


def test_calc_card_inherits_weak_pit(repo, policy):
    cur = _filing([_metric("营业收入", "OPERATE_INCOME", 91 * B, "2026-06-30")], pit="weak")
    prior = _filing([_metric("营业收入", "OPERATE_INCOME", 90 * B, "2025-06-30")],
                    published="2025-08-13", pit="weak")
    cards = _derive(repo, policy, [cur, prior])
    assert cards[0].pit == "weak" and any("追溯调整" in c for c in cards[0].caveats)


# ---------- 核验器按口径筛候选 ----------

def test_verifier_respects_cumulative_and_single_quarter():
    card = _filing([
        _metric("营业收入同比(累计)", "OPERATE_INCOME_YOY", "1.11", "2026-06-30", unit="%", basis="change"),
        _metric("营业收入(单季)同比(单季)", "OPERATE_INCOME_SQ_YOY", "-7.50", "2026-06-30", unit="%", basis="change"),
    ])
    ev = {card.id: card}

    def status(text):
        return verify_claim(Claim(id="c", text=text, kind="fact", evidence_ids=[card.id]), ev, AS_OF).status

    assert status("2026年上半年营业收入累计同比下降7.50%。") == "disputed"   # 单季数不能给累计背书
    assert status("2026年第二季度单季营业收入同比上升1.11%。") == "disputed"
    assert status("2026年上半年营业收入累计同比上升1.11%。") == "verified"
    assert status("2026年第二季度单季营业收入同比下降7.50%。") == "verified"


# ---------- 先取数再计划：搜索词用已核验公司名 ----------

class _WrongNameLLM(FakeLLM):
    def json(self, role, system, user, schema, max_tokens=None):
        if schema is IntakeOut:
            return IntakeOut(company_name="五粮液", ticker=None), 30
        if schema is QueryPlan:
            m = re.search(r"公司：(.+?)（\d{6}）", user)
            name = m.group(1) if m else "未知"
            return QueryPlan(queries=[f"{name} 业绩", f"{name} 行业库存", f"{name} 风险"]), 50
        return super().json(role, system, user, schema, max_tokens)


def test_plan_uses_verified_company_name(repo, policy):
    from tests.test_phase2 import make_service, run
    svc = make_service(repo, policy, _WrongNameLLM())
    tid, task = run(svc)
    assert task["status"] == "done", task
    plan = repo.list_working(tid)
    queries = next(w["data"] for w in plan if w["kind"] == "plan")
    assert any("示例酒业" in q for q in queries) and not any("五粮液" in q for q in queries)
