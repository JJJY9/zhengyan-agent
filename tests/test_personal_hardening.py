"""个人情境官与约束回归：汇总逻辑、期限缺数据、匹配强度、影响基线、文案、精度、新鲜度、约束校验。"""
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from advisor.contracts import Claim, Dossier, Metric, QualityResult, ResearchRequest
from advisor.evidence import EvidenceFactory
from advisor.personal import Holding, ProfileInput, ProfileVersion, RiskProfile, assess
from advisor.services import Services
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.watch.models import Alert
from langgraph.checkpoint.memory import InMemorySaver
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools

NOW = datetime.now(timezone.utc)


def _dossier(evidence_ids=(), claims=(), status="committed", unknowns=(), dissents=()):
    return Dossier(key=CODE, version=1, ticker=CODE, name="示例酒业", as_of=datetime(2026, 10, 18).date(),
                   question="q", route="standard", status=status, claims=list(claims),
                   evidence_ids=list(evidence_ids), unknowns=list(unknowns), dissents=list(dissents),
                   quality=QualityResult(decision="pass"), task_id="t", created_at=NOW)


def _profile(**risk):
    return ProfileVersion(version=1, risk=RiskProfile(**risk), holdings=[], confirmed_by="owner",
                          confirmed_at=NOW)


def _market(metrics):
    f = EvidenceFactory(lambda h, c: None)
    return f.make("m", level="A", source_type="market", title="行情", origin_id="market:x", as_of=datetime(2026, 10, 18).date(),
                  excerpt="", metrics=metrics, skill="market.daily", skill_version="1")


def _check(fit, key):
    return next(c for c in fit.checks if c.key == key)


# ---------- 汇总逻辑 ----------

def test_summary_without_confirmed_constraints():
    d = _dossier([], [Claim(id="c1", text="无关结论", kind="fact", status="verified")])
    assert assess(d, [], None).summary == "缺少你的确认，情境官无法判断"
    assert assess(d, [], _profile()).summary == "缺少你的确认，情境官无法判断"   # 确认了空档案也一样


# ---------- 期限缺数据 ----------

def test_horizon_missing_vol_is_unknown():
    card = _market([Metric(name="收盘价(前复权)", field="close", value=Decimal(10), unit="元",
                           period="2026-10-16", basis="point")])
    fit = assess(_dossier([card.id]), [card], _profile(horizon_months=9, max_drawdown_pct=20))
    horizon = _check(fit, "horizon")
    assert horizon.status == "unknown" and "没有近 60 日波动" in horizon.detail


# ---------- 购买理由匹配强度 ----------

def test_thesis_requires_two_shared_bigrams():
    claim = Claim(id="c1", text="公司当前估值处于历史高位，存在回落风险", kind="inference", status="verified")
    d = _dossier([], [claim])
    profile = ProfileVersion(version=1, risk=RiskProfile(), holdings=[Holding(ticker=CODE, thesis="估值便宜")],
                             confirmed_by="owner", confirmed_at=NOW)
    thesis = _check(assess(d, [], profile), "thesis")
    assert thesis.status != "ok" and "暂未见反证" not in thesis.detail   # 单个通用词不能把相反结论算支持


# ---------- 回撤展示精度 ----------

def test_drawdown_uses_display_precision():
    c20 = _market([Metric(name="近250日最大回撤", field="mdd_250d", value=Decimal("-20.04"), unit="%",
                          period="2026-10-16", basis="change")])
    fit20 = assess(_dossier([c20.id]), [c20], _profile(max_drawdown_pct=20))
    assert _check(fit20, "drawdown").status == "attention" and "20.0%" in _check(fit20, "drawdown").detail
    c21 = _market([Metric(name="近250日最大回撤", field="mdd_250d", value=Decimal("-20.06"), unit="%",
                          period="2026-10-16", basis="change")])
    fit21 = assess(_dossier([c21.id]), [c21], _profile(max_drawdown_pct=20))
    assert _check(fit21, "drawdown").status == "conflict" and "20.1%" in _check(fit21, "drawdown").detail


# ---------- 结论完整度文案 ----------

def test_evidence_wording_matches_status():
    committed = assess(_dossier(unknowns=["渠道库存"]), [], None)
    check = _check(committed, "evidence")
    assert check.status == "attention" and "已通过质量门控" in check.detail and "部分结论" not in check.detail
    partial = _dossier(status="partial", claims=[Claim(id="c", text="x", kind="fact", status="disputed")])
    detail2 = _check(assess(partial, [], None), "evidence").detail
    assert "部分结论" in detail2 and "存疑/驳回 1" in detail2


# ---------- 新鲜度 ----------

def test_freshness_never_scanned_and_reviewing_alerts():
    fit = assess(_dossier(), [], None, subscribed=True, open_alerts=[], scan_note="", scan_at=None)
    assert _check(fit, "freshness").status == "unknown" and "还没有扫描过" in _check(fit, "freshness").detail
    fit2 = assess(_dossier(), [], None, subscribed=True, open_alerts=[("al_1", "价格异动")], scan_at="2026-11-25")
    assert _check(fit2, "freshness").status == "attention"      # 复核中的提醒也会传入（API 负责）


def test_fit_endpoint_counts_reviewing_alert(repo, policy):
    from fastapi.testclient import TestClient
    from advisor.api.app import create_app
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    svcs = Services(deps, ResearchService(deps, InMemorySaver()))
    svcs.research.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=datetime(2026, 10, 18).date(),
                                        route="standard"), background=False)
    store = svcs.watch.store
    store.subscribe(CODE, "示例酒业")
    store.mark_scanned(CODE, "有新变化")
    alert = Alert(id="al_x", ticker=CODE, name="示例酒业", dossier_key=CODE, dossier_version=1,
                  fingerprint="fp", events=[], affected=[], created_at=NOW)
    store.add_alert(alert)
    store.try_start_review(alert.id)                             # status=reviewing
    with TestClient(create_app(deps=deps)) as c:
        fit = c.get(f"/api/profile/fit/{CODE}").json()["fit"]
    fresh = next(x for x in fit["checks"] if x["key"] == "freshness")
    assert fresh["status"] == "attention"                        # 复核中的提醒不能被当成不存在


# ---------- 影响基线 ----------

def test_profile_impact_uses_real_parent_version(repo, policy):
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools())
    svcs = Services(deps, ResearchService(deps, InMemorySaver()))
    card = deps.factory.make("m", level="A", source_type="market", title="行情", origin_id="market:x",
                  as_of=datetime(2026, 10, 18).date(), excerpt="",
                  metrics=[Metric(name="近250日最大回撤", field="mdd_250d", value=Decimal("-40"), unit="%",
                                  period="2026-10-16", basis="change")], skill="market.daily", skill_version="1")
    repo.put_evidence(card)
    repo.db.dossiers.insert_one(_dossier([card.id]).model_dump(mode="json"))

    svcs.confirm_profile(ProfileInput(risk=RiskProfile(max_drawdown_pct=90)), "owner", True)   # v1
    orig = svcs.profiles.confirm

    def racing(data, actor, confirmed):
        payload = data.model_dump()
        payload.update(risk=RiskProfile(max_drawdown_pct=5).model_dump(), holdings=[])
        v2 = ProfileVersion(**payload, version=2, parent_version=1, confirmed_by="other", confirmed_at=NOW)
        svcs.profiles.col.insert_one(v2.model_dump(mode="json"))   # 另一标签页抢先写入
        return orig(data, actor, confirmed)

    svcs.profiles.confirm = racing                                  # type: ignore[method-assign]
    svcs.confirm_profile(ProfileInput(risk=RiskProfile(max_drawdown_pct=5)), "owner", True)    # v3(parent=2)

    v3 = svcs.profiles.current()
    assert v3.version == 3 and v3.parent_version == 2
    assert v3.impact == []                                          # v2→v3 没有变化，不能按 v1 基线误报


# ---------- 约束校验 ----------

def test_holding_thesis_must_be_meaningful():
    with pytest.raises(ValidationError):
        Holding(ticker=CODE, thesis="    ")
    with pytest.raises(ValidationError):
        Holding(ticker=CODE, thesis="。。。。")
    assert Holding(ticker=CODE, thesis="  估值便宜  ").thesis == "估值便宜"
