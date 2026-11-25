"""版本差异与依赖传播：新版本不覆盖旧版本，而是先展示变化与影响。

依赖图用显式关系实现：主张 → 证据 → 报告期。最新报告期前移后，依赖旧报告期的旧结论标记"待复核"。
"""
from difflib import SequenceMatcher

from advisor.contracts import Claim, Dossier, EvidenceCard
from advisor.evidence.facts import report_periods
from advisor.evidence.grading import cst_date

SIM_THRESHOLD = 0.8        # 主张配对阈值：同模板不同指标（相似度约 0.67）不能被当成同一条结论
CLAIM_KINDS = ("fact", "inference")


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _calc_metrics(d: Dossier, ev: dict[str, EvidenceCard]) -> dict[str, dict]:
    """指标名 → 最新报告期的那一条：同一张卡里本期与去年同期可能同名，取报告期更晚的。"""
    out = {}
    for i in d.evidence_ids:
        card = ev.get(i)
        if card and card.source_type in ("calc", "market"):
            for m in card.metrics:
                prev = out.get(m.name)
                if prev is None or m.period >= prev["period"]:
                    out[m.name] = {"value": str(m.value), "unit": m.unit, "period": m.period[:10], "basis": m.basis}
    return out


def _pair_claims(new: list[Claim], old: list[Claim]) -> tuple[list[dict], list[str], list[str], list[dict]]:
    """一对一配对主张：一条旧主张最多对应一条新主张，按相似度从高到低贪心消费。

    返回（保留、新增、不再出现、核验状态变化）。配对覆盖所有 fact/inference 主张（含未通过核验的），
    这样「verified → disputed」会显示为状态变化，而不是「结论不再出现」。
    """
    new_c = [c for c in new if c.kind in CLAIM_KINDS and c.text.strip()]
    old_c = [c for c in old if c.kind in CLAIM_KINDS and c.text.strip()]
    pairs = []
    for ni, c in enumerate(new_c):
        for oi, o in enumerate(old_c):
            sim = _similar(o.text, c.text)
            if sim >= SIM_THRESHOLD:
                pairs.append((sim, oi, ni))
    pairs.sort(reverse=True)
    used_old, used_new = set(), set()
    kept, status_changed = [], []
    for sim, oi, ni in pairs:
        if oi in used_old or ni in used_new:
            continue
        used_old.add(oi)
        used_new.add(ni)
        o, c = old_c[oi], new_c[ni]
        if o.status != c.status:
            status_changed.append({"text": c.text, "old_status": o.status, "new_status": c.status,
                                   "similarity": round(sim, 2)})
        elif c.status == "verified":
            kept.append({"old": o.text, "new": c.text, "similarity": round(sim, 2)})
    added = [c.text for ni, c in enumerate(new_c) if ni not in used_new]
    removed = [o.text for oi, o in enumerate(old_c) if oi not in used_old]
    return kept, added, removed, status_changed


def diff_dossiers(new: Dossier, old: Dossier, ev: dict[str, EvidenceCard]) -> dict:
    om, nm = _calc_metrics(old, ev), _calc_metrics(new, ev)
    metrics = [
        {"name": k, "old": om.get(k), "new": nm.get(k),
         "changed": (om.get(k) or {}).get("value") != (nm.get(k) or {}).get("value")
         or (om.get(k) or {}).get("unit") != (nm.get(k) or {}).get("unit")}
        for k in dict.fromkeys(list(nm) + list(om))
    ]
    kept, added, removed, status_changed = _pair_claims(new.claims, old.claims)

    latest_new = max(report_periods([i for i in new.evidence_ids], ev), default="")
    stale = []
    for c in old.claims:
        if c.status != "verified":
            continue
        periods = report_periods(c.evidence_ids, ev)
        if periods and latest_new and max(periods) < latest_new:
            stale.append({"text": c.text, "reason": f"依赖的报告期 {max(periods)} 已被 {latest_new} 取代，需复核"})

    old_ident = {(c.origin_id, c.snapshot_hash) for i in old.evidence_ids if (c := ev.get(i))}
    evidence_added = []
    for i in new.evidence_ids:
        c = ev.get(i)
        if c is None or c.source_type not in ("filing", "announcement", "web"):
            continue
        if (c.origin_id, c.snapshot_hash) in old_ident:      # 同一来源同一快照已在旧版本，卡 id 变了也不算新增
            continue
        day = cst_date(c.published_at) if c.published_at else None
        if day is not None and day < old.as_of:              # 早于旧版本研究时点的证据不算「新增」
            continue
        evidence_added.append({"id": c.id, "title": c.title, "level": c.level, "source_type": c.source_type,
                               "date": day.isoformat() if day else None})
    return {
        "from_version": old.version, "to_version": new.version,
        "as_of": {"old": old.as_of.isoformat(), "new": new.as_of.isoformat()},
        "route": {"old": old.route, "new": new.route},
        "status": {"old": old.status, "new": new.status},
        "metrics": metrics,
        "claims": {"kept": kept, "added": added, "removed": removed, "status_changed": status_changed},
        "stale_claims": stale,
        "evidence_added": evidence_added,
        "unknowns": {"resolved": [u for u in old.unknowns if u not in new.unknowns],
                     "new": [u for u in new.unknowns if u not in old.unknowns]},
        "triggers": {"old": [t.condition for t in old.review_triggers], "new": [t.condition for t in new.review_triggers]},
        "prior_review": [p.model_dump(mode="json") for p in new.prior_review],
    }
