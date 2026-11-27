"""版本差异、对照评测、质量指标与持续更新评测的回归。"""
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

from advisor.contracts import Claim, Dossier, Metric, QualityResult, ResearchRequest
from advisor.evals.compare import run_compare, summarize
from advisor.evals.metrics import dossier_metrics
from advisor.evals.update import score_case
from advisor.evidence.factory import EvidenceFactory
from advisor.memory.diff import diff_dossiers


def _claim(cid, text, status="verified", kind="inference"):
    return Claim(id=cid, text=text, kind=kind, evidence_ids=[], status=status)


def _dossier(version, as_of, claims=None, evidence_ids=None):
    return Dossier(key="600999", version=version, ticker="600999", name="示例酒业", as_of=as_of,
                   question="q", route="standard", status="committed", claims=claims or [],
                   evidence_ids=evidence_ids or [], quality=QualityResult(decision="pass"),
                   task_id="t", created_at=datetime.now(timezone.utc))


def _card(content="c", **kw):
    f = EvidenceFactory(lambda h, c: None)
    fields = dict(level="A", source_type="calc", title="t", origin_id="calc:x", as_of=date(2026, 6, 30),
                  excerpt="e", skill="calc", skill_version="1")
    return f.make(content, **(fields | kw))


def _metric(name, value, period):
    return Metric(name=name, field="f", value=Decimal(value), unit="%", period=period, basis="ratio")


# ---------- diff：同名指标取最新期 ----------

def test_diff_metrics_pick_latest_period():
    old_card = _card("old", metrics=[_metric("归母净利率(累计)", "10.00", "2026-06-30"),
                                      _metric("归母净利率(累计)", "8.00", "2025-06-30")])
    new_card = _card("new", as_of=date(2026, 9, 30),
                     metrics=[_metric("归母净利率(累计)", "6.00", "2026-09-30"),
                              _metric("归母净利率(累计)", "8.00", "2025-09-30")])
    old = _dossier(1, date(2026, 6, 30), evidence_ids=[old_card.id])
    new = _dossier(2, date(2026, 9, 30), evidence_ids=[new_card.id])
    diff = diff_dossiers(new, old, {old_card.id: old_card, new_card.id: new_card})
    row = next(m for m in diff["metrics"] if m["name"] == "归母净利率(累计)")
    assert row["old"]["value"] == "10.00" and row["new"]["value"] == "6.00" and row["changed"] is True


# ---------- diff：主张一对一配对 ----------

def test_diff_claims_one_to_one_and_template_split():
    old = _dossier(1, date(2026, 6, 30), claims=[
        _claim("c1", "第二季度单季营业收入370.00亿元。"),
        _claim("c2", "2026年上半年营业收入910.00亿元，累计同比1.11%。"),
    ])
    new = _dossier(2, date(2026, 9, 30), claims=[
        _claim("c2", "2026年上半年营业收入910.00亿元，累计同比1.11%。"),
        _claim("c3", "第二季度单季归母净利润445.00亿元。"),      # 与 c1 同模板不同指标：相似度约 0.62，不能被当成同一条
    ])
    diff = diff_dossiers(new, old, {})
    claims = diff["claims"]
    assert len(claims["kept"]) == 1 and claims["kept"][0]["new"] == new.claims[0].text
    assert claims["added"] == ["第二季度单季归母净利润445.00亿元。"]
    assert claims["removed"] == ["第二季度单季营业收入370.00亿元。"]


def test_diff_claims_do_not_double_keep():
    old = _dossier(1, date(2026, 6, 30), claims=[_claim("a", "公司毛利率保持稳定。"), _claim("b", "公司毛利率保持稳定。")])
    new = _dossier(2, date(2026, 9, 30), claims=[_claim("a", "公司毛利率保持稳定。")])
    claims = diff_dossiers(new, old, {})["claims"]
    assert len(claims["kept"]) == 1 and claims["removed"] == ["公司毛利率保持稳定。"]


def test_diff_status_change_is_not_removal():
    text = "公司上半年毛利率为30%。"
    old = _dossier(1, date(2026, 6, 30), claims=[_claim("a", text, status="verified")])
    new = _dossier(2, date(2026, 9, 30), claims=[_claim("a", text, status="disputed")])
    claims = diff_dossiers(new, old, {})["claims"]
    assert claims["removed"] == [] and claims["added"] == []
    assert claims["status_changed"] == [{"text": text, "old_status": "verified", "new_status": "disputed", "similarity": 1.0}]


# ---------- diff：新增证据的时区与身份 ----------

def test_diff_evidence_uses_beijing_day():
    card = _card("web", level="B", source_type="web", origin_id="example.com/a",
                 published_at=datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc))     # 北京 10-19 01:00
    old = _dossier(1, date(2026, 10, 18))
    new = _dossier(2, date(2026, 10, 19), evidence_ids=[card.id])
    added = diff_dossiers(new, old, {card.id: card})["evidence_added"]
    assert added == [{"id": card.id, "title": "t", "level": "B", "source_type": "web", "date": "2026-10-19"}]


def test_diff_evidence_identity_not_card_id():
    old_card = _card("same", level="B", source_type="web", origin_id="example.com/a", as_of=date(2026, 10, 18))
    new_card = _card("same", level="B", source_type="web", origin_id="example.com/a", as_of=date(2026, 11, 1))
    assert old_card.id != new_card.id                     # 卡 id 含 as_of，跨版本必然不同
    old = _dossier(1, date(2026, 10, 18), evidence_ids=[old_card.id])
    new = _dossier(2, date(2026, 11, 1), evidence_ids=[new_card.id])
    added = diff_dossiers(new, old, {old_card.id: old_card, new_card.id: new_card})["evidence_added"]
    assert added == []                                    # 同一来源同一快照：不是新增证据


# ---------- 对照评测：失败用例不被丢掉 ----------

def test_summarize_marks_failed_cases_insufficient():
    m = {"score": 0.7, "tokens": 100}
    results = [{"route": "standard", "metrics": m}, {"route": "standard", "metrics": m},
               {"route": "deep", "metrics": m}, {"route": "deep", "error": "boom"}]
    s = summarize(results)
    assert s["verdict"]["decision"] == "insufficient" and s["verdict"]["failed_cases"] == 1
    assert s["routes"]["deep"]["score_all"] == 0.35      # 失败用例按 0 分计入有效分


class _CompareSvc:
    def __init__(self, repo):
        self.deps = SimpleNamespace(repo=repo)
        self.resumed = []
        self.research = SimpleNamespace(resume=self.resume)

    def start(self, req: ResearchRequest, background=False, namespace=None, **kw):
        self.deps.repo.create_task("t_x", req.model_dump(mode="json") | {"namespace": namespace})
        self.deps.repo.update_task("t_x", status="waiting_approval")
        return "t_x"

    def resume(self, tid, approve, background=False, **kw):
        self.resumed.append((tid, approve))
        self.deps.repo.update_task(tid, status="failed", error="评测内已自动批准后结束")


def test_compare_auto_approves_waiting_task(repo):
    svc = _CompareSvc(repo)
    run_compare(svc, [{"id": "c1", "question": "复核 600999 半年报业绩"}], ["deep"], run_id="ev_x")
    assert svc.resumed == [("t_x", True)]
    run = repo.get_eval("ev_x")
    assert run["status"] == "done" and run["results"][0]["auto_approved"] is True
    assert "waiting_approval" not in (run["results"][0].get("error") or "")


# ---------- 质量指标：N/A 与假设排除 ----------

def test_metrics_numeric_na_and_assumptions_excluded():
    d = _dossier(1, date(2026, 6, 30), claims=[
        _claim("a", "公司上半年收入同比增长。", status="verified", kind="fact"),
        _claim("b", "公司渠道库存下降。", status="verified", kind="inference"),
        _claim("c", "公司发布公告说明分红安排。", status="verified", kind="fact"),
    ])
    m = dossier_metrics(d, {}, {"result": {}})
    assert m["numeric_claims"] == 0 and m["numeric_accuracy"] is None

    d2 = _dossier(1, date(2026, 6, 30), claims=[
        _claim("a", "预计下半年营业收入 120 亿元。", status="rejected", kind="fact"),
        _claim("b", "假设原材料价格保持稳定。", status="verified", kind="assumption"),
    ])
    m2 = dossier_metrics(d2, {}, {"result": {}})
    assert m2["verified_rate"] == 0.0 and m2["assumptions"] == 1 and m2["numeric_accuracy"] == 0.0


# ---------- 持续更新评测：一对一匹配 ----------

def test_update_match_is_one_to_one():
    alert = {"id": "a1", "ticker": "600999", "name": "示例酒业", "events": [],
             "affected": [{"claim_id": "c1"}, {"claim_id": "c2"}]}
    old = {"version": 1, "as_of": "2026-06-30", "claims": [
        {"id": "c1", "text": "经营现金流为正"},
        {"id": "c2", "text": "经营现金流为正且覆盖净利润"},
    ]}
    new = {"version": 2, "as_of": "2026-09-30", "claims": [
        {"id": "c2", "text": "经营现金流为正且覆盖净利润"},
    ], "prior_review": [
        {"prior_claim": "经营现金流为正且覆盖净利润", "verdict": "retained", "reason": "复核", "evidence_ids": []},
    ]}
    case = score_case(alert, old, new, {}, {})
    rows = {r["id"]: r for r in case["claims"]}
    assert case["metrics"]["coverage"] == 0.5                       # 一条复核只算一条结论的交代
    assert case["metrics"]["decisive"] == 1.0                       # 有交代的那条给出了明确判定
    assert rows["c1"]["verdict"] is None and rows["c1"]["kept"] is False
    assert rows["c2"]["verdict"] == "retained" and rows["c2"]["kept"] is True
