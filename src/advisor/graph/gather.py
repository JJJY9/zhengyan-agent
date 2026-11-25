"""搜索子系统流水线（单 Agent 与各专家共用）：搜索 → 过滤 → 聚类去转载 → 白名单抓取 → 清洗 → 分级 → 快照。"""
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from advisor.contracts import EvidenceCard, SearchHit
from advisor.evidence import assign_clusters, extract_publisher, grade_web, origin_id, sanitize
from advisor.harness import BudgetExceeded, HarnessDenied, SkillFailed
from advisor.skills.data.web import fetch_allowlist, host_matches


@dataclass
class GatherResult:
    cards: list[EvidenceCard] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    hits: int = 0
    clusters: int = 0
    fetched: int = 0


def gather_web(h, deps, queries: list[str], as_of: date, route: str, max_fetch: int,
               log: Callable[[str], None] = lambda m: None, search_cfg: dict | None = None) -> GatherResult:
    """search_cfg 给定时（任务级 RSI 快照）检索参数与追加站点都按它来，不再实时读在用版本。"""
    cfg = search_cfg if search_cfg is not None else (deps.rsi.search() if deps.rsi else deps.policy.search)
    pin = {"rerank_min": float(cfg["rerank_min"]) if "rerank_min" in cfg else None,
           "site_allow_add": [s for s in cfg.get("fetch_allow_add") or []]} if search_cfg is not None else {}
    res = GatherResult()
    hits: list[SearchHit] = []
    for q in queries:
        try:
            out = h.call("web.search", query=q, as_of=as_of, count=10,
                         lookback_days=int(cfg.get("lookback_days", 120)), **pin)
            hits.extend(out.hits)
            res.searched.append(q)
        except BudgetExceeded as e:
            res.unknowns.append(f"搜索预算耗尽，未执行：{q}")
            log(str(e))
            break
        except (SkillFailed, HarnessDenied) as e:
            res.unknowns.append(f"搜索失败：{q}（{e}）")
    unique: dict[str, SearchHit] = {}
    for hit in hits:
        unique.setdefault(hit.url, hit)      # 多查询命中同一篇只算一次，转载计数不虚高
    hits = list(unique.values())
    clusters = assign_clusters(hits)
    by_cluster: dict[str, list[SearchHit]] = {}
    for hit in hits:
        by_cluster.setdefault(clusters[hit.url], []).append(hit)
    b_min = int(cfg.get("authority_b_min", 3))
    can_fetch = deps.policy.is_allowed(route, "web.fetch")
    allow = fetch_allowlist(deps.policy, cfg)

    def _rank(x: SearchHit) -> tuple[int, float]:
        return -x.authority, -x.rerank_score

    # 代表决定「抓哪篇、按哪篇定级」：优先选有日期且在白名单内的成员，
    # 否则权威度再高但抓不到/无日期的成员会把整簇压成 C。
    reps: dict[str, SearchHit] = {}
    for cid, members in by_cluster.items():
        candidates = sorted(members, key=_rank)
        if can_fetch:
            fetchable = [m for m in candidates
                         if m.authority >= b_min and m.date is not None and host_matches(m.url, allow)]
            candidates = fetchable or candidates
        reps[cid] = candidates[0]
    sizes = {cid: len(members) for cid, members in by_cluster.items()}
    to_fetch = [
        hit for hit in reps.values()
        if can_fetch and hit.authority >= b_min and hit.date is not None and host_matches(hit.url, allow)
    ][:max_fetch]
    fetched: dict[str, str] = {}
    fetch_pin = {"site_allow_add": pin["site_allow_add"]} if search_cfg is not None else {}
    for hit in to_fetch:
        try:
            fetched[hit.url] = h.call("web.fetch", url=hit.url, **fetch_pin).content
        except BudgetExceeded:
            break
        except (SkillFailed, HarnessDenied) as e:
            log(f"一篇网页原文没读到（{hit.url}），跳过：{e}")
    for cid, hit in reps.items():
        cleaned_raw, sus_raw = sanitize(fetched.get(hit.url) or "", max_chars=20000)
        if cleaned_raw:
            content, fetched_ok = cleaned_raw, True          # 清洗后仍有正文才算抓到了快照
        else:
            content, sus_fb = sanitize(hit.summary or hit.snippet or "", max_chars=20000)
            fetched_ok = False
            sus_raw = sus_raw or sus_fb
        excerpt, sus_e = sanitize(hit.summary or hit.snippet or content, max_chars=800)
        title, sus_t = sanitize(hit.title or "", max_chars=200)
        level, caveats = grade_web(hit, fetched_ok, as_of, b_min)
        if sus_raw or sus_e or sus_t:
            level = "C"
            caveats.append("检测到疑似指令文本，已剔除并降为线索")
        if sizes[cid] > 1:
            caveats.append(f"同一事件另有 {sizes[cid] - 1} 篇相似/转载报道，未计为独立证据")
        card = deps.factory.make(
            content=content, level=level, source_type="web", title=title or "(标题不可用)", source_url=hit.url,
            site_name=hit.site_name, origin_id=origin_id(hit.url),
            origin_publisher=extract_publisher(hit.summary, hit.site_name), cluster_id=cid,
            published_at=hit.date, as_of=as_of, pit="weak", excerpt=excerpt, authority=hit.authority,
            rerank_score=hit.rerank_score, caveats=caveats, skill="web.search", skill_version="1.0.0",
        )
        deps.repo.put_evidence(card)
        res.cards.append(card)
    res.hits, res.clusters, res.fetched = len(hits), len(reps), len(fetched)
    return res
