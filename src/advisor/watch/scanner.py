"""订阅扫描与增量复核。

扫描：以今天为时点重新取财报、公告、行情，与档案当时的证据对比，找出「前提变化」：
新报告期、档案时点之后的公告、超过阈值的价格变化、被触发的复核条件。
依赖传播：主张 → 证据 → 报告期 / 行情时点。只把受影响的旧结论列为「待复核」，档案本身不改。
增量复核：带着受影响的结论发起一次标准研究，生成新版本，并在报告里逐条复核旧结论。
"""
import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from advisor.contracts import Dossier, EvidenceCard, ResearchRequest
from advisor.evidence import fact_value, metric_facts, report_periods
from advisor.graph.deps import Deps
from advisor.harness import Budget, BudgetExceeded, HarnessDenied, SkillFailed, TaskCancelled
from advisor.skills.compute.finance import normalize_period
from advisor.watch.models import EVENT_LABEL, AffectedClaim, Alert, WatchEvent
from advisor.watch.store import WatchStore

TRIGGER_WORDS = {
    "new_report": ("财报", "季报", "年报", "半年报", "业绩", "报告期", "营收", "利润"),
    "announcement": ("公告", "回购", "减持", "增持", "分红", "监管", "处罚", "诉讼", "重组"),
    "price_move": ("股价", "价格", "估值", "PE", "市盈率", "回撤", "下跌", "上涨"),
}
PRICE_SOURCES = ("market.daily", "valuation.snapshot")
ANNOUNCE_WINDOWS = 5              # 公告回看最多 5×400 天，覆盖大多数档案时点，避免窗口滑动重复提醒
RECOVER_AGE_S = 1800              # 复核任务停滞超过此时长视为僵尸（≥ 深研 1800s 总时限）


def _norm_period(period: str) -> str:
    """报告期归一（兼容 YYYYMMDD），与 compute/ttm 的口径一致。"""
    return normalize_period(period) or str(period)[:10]


def _rebase_close(repo, card: EvidenceCard, day: str) -> Decimal | None:
    """今日行情卡的原始行里，档案交易日的复权收盘价：同一复权基准才能比较。"""
    try:
        rows = json.loads(repo.get_snapshot(card.snapshot_hash) or "")
    except (TypeError, ValueError):
        return None
    for r in rows:
        if str(r.get("date"))[:10] == day and r.get("close") is not None:
            try:
                return Decimal(str(r["close"]))
            except (InvalidOperation, ValueError):
                return None
    return None


class WatchError(ValueError):
    pass


class Watcher:
    def __init__(self, deps: Deps, store: WatchStore):
        self.deps = deps
        self.store = store

    # ---------- 扫描 ----------
    def scan(self, ticker: str, today: date | None = None) -> Alert | None:
        """有变化返回提醒（同一批变化只提醒一次），没有变化返回 None。"""
        today = today or date.today()
        repo = self.deps.repo
        # 当前档案按研究时点取：CLI 回填会产出「版本号更高、as_of 更早」的档案，
        # 按版本号取会把回填的旧结论当最新，误报「新财报」并错误关闭真实提醒
        d = repo.current_dossier(ticker, today)
        if d is None:
            raise WatchError(f"{ticker} 还没有档案，先做一次研究再订阅")
        self._supersede_open_alerts(d)          # 新版本已覆盖的旧提醒自动关闭，避免同批提醒长期 open
        sub = self.store.get_sub(ticker)
        threshold = sub.price_move_pct if sub else 10
        evidence = {c.id: c for c in repo.get_evidence(d.evidence_ids + [i for c in d.claims for i in c.evidence_ids])}
        h = self.deps.harness(f"watch_{ticker}", "quick", today, Budget(limits=self.deps.policy.route("quick").budget),
                              node="watch", principal="watcher")
        events, affected, notes = [], {}, []
        for detect in (self._new_report, self._announcements, self._price_move):
            try:
                found = detect(h, d, evidence, today, threshold)
            except (SkillFailed, HarnessDenied, BudgetExceeded) as e:
                notes.append(f"{detect.__name__.strip('_')} 取数失败：{e}")
                continue
            if found:
                event, claims, cards = found
                events.append(event)
                for c in cards:
                    repo.put_evidence(c)
                for a in claims:
                    affected.setdefault(a.claim_id, a)
        events += self._triggers(d, {e.kind for e in events})
        if not events:
            self.store.mark_scanned(ticker, "无变化" + (f"（{'；'.join(notes)}）" if notes else ""))
            return None
        alert, created = self.store.add_alert(Alert(
            id="al_" + uuid.uuid4().hex[:8], ticker=ticker, name=d.name, dossier_key=d.key, dossier_version=d.version,
            fingerprint=_fingerprint(d, events), events=events, affected=list(affected.values()),
            created_at=datetime.now(timezone.utc), note="；".join(notes),
        ))
        note = alert_summary(alert, created) + (f"（{'；'.join(notes)}）" if notes else "")
        self.store.mark_scanned(ticker, note)
        return alert

    def _supersede_open_alerts(self, d: Dossier) -> None:
        """档案已有更新的「研究时点」版本：旧版本触发的待处理提醒自动关闭（复核完成后 open_alerts 能清零）。"""
        versions = {v["version"]: str(v.get("as_of"))[:10] for v in self.deps.repo.list_dossier_versions(d.ticker)}
        current = (str(d.as_of)[:10], d.version)
        for a in self.store.list_alerts(d.ticker, "open"):
            if (versions.get(a.dossier_version, ""), a.dossier_version or 0) < current:
                self.store.update_alert(a.id, status="dismissed",
                                        note=f"已有更新的档案 v{d.version}（时点 {d.as_of}），旧提醒自动关闭")

    def scan_all(self, today: date | None = None, log=lambda m: None,
                 is_cancelled=lambda: False) -> list[Alert]:
        self.sync()                             # 定时/CLI 扫描也回写复核结果（不再只在打开网页时同步）
        out = []
        for sub in self.store.list_subs():
            if is_cancelled():                  # 取消检查点：每个订阅之间
                raise TaskCancelled("扫描作业已取消")
            try:
                alert = self.scan(sub.ticker, today)
            except WatchError as e:
                log(f"{sub.ticker}：{e}")
                continue
            note = (self.store.get_sub(sub.ticker) or sub).last_scan_note
            log(f"{sub.name or sub.ticker}（{sub.ticker}）：{alert_summary(alert) if alert else (note or '没有变化，不提醒')}")
            if alert:
                out.append(alert)
        return out

    def _new_report(self, h, d: Dossier, evidence: dict[str, EvidenceCard], today: date, _):
        st = h.call("financials.statements", code=d.ticker, as_of=today, periods=5)
        latest_new = max((_norm_period(m.period) for c in st.cards for m in c.metrics), default="")
        latest_old = max((_norm_period(p) for p in report_periods(d.evidence_ids, evidence)), default="")
        if not latest_old or latest_new <= latest_old:
            return None
        card = next(c for c in st.cards if any(_norm_period(m.period) == latest_new for m in c.metrics))
        claims = [AffectedClaim(claim_id=c.id, text=c.text, reason=f"依赖的报告期 {max(_norm_period(x) for x in p)} 已有更新的 {latest_new}")
                  for c in d.claims if c.status == "verified" and (p := report_periods(c.evidence_ids, evidence))
                  and max(_norm_period(x) for x in p) < latest_new]
        event = WatchEvent(kind="new_report", title=f"{latest_new} 报告期财报已披露", date=card.published_at.date().isoformat()
                           if card.published_at else None, detail=f"档案用的是 {latest_old}", evidence_ids=[card.id])
        return event, claims, [card]

    def _announcements(self, h, d: Dossier, evidence, today: date, _):
        """新公告以「是否已进入档案证据」判定：与 as_of 同日、但研究之后发布的公告也能提醒。"""
        known = {c.id for c in evidence.values() if c.source_type == "announcement"} | \
                {c.origin_id for c in evidence.values() if c.source_type == "announcement"}
        found: dict[str, EvidenceCard] = {}
        cursor = today
        for _ in range(ANNOUNCE_WINDOWS):
            days = min(max((cursor - d.as_of).days, 7), 400)
            out = h.call("announcements.list", code=d.ticker, as_of=cursor, lookback_days=days)
            for c in out.cards:
                if not c.published_at or c.published_at.date() < d.as_of:
                    continue
                if c.id in known or (c.origin_id and c.origin_id in known):
                    continue
                found[c.origin_id or c.id] = c
            if (cursor - d.as_of).days <= 400:
                break
            cursor = cursor - timedelta(days=400)
        new = sorted(found.values(), key=lambda c: c.published_at)
        if not new:
            return None
        titles = "；".join(c.title for c in new[:3])
        event = WatchEvent(kind="announcement", title=f"档案时点后新增 {len(new)} 条公告", detail=titles,
                           date=max(c.published_at.date() for c in new).isoformat(), evidence_ids=[c.id for c in new])
        return event, [], new

    def _price_move(self, h, d: Dossier, evidence, today: date, threshold: float):
        old_metric = next((m for c in evidence.values() if c.skill == "market.daily"
                           for m in c.metrics if m.field == "close"), None)
        if old_metric is None:
            return None
        old_day = str(old_metric.period)[:10]
        try:
            lookback = min(max((today - date.fromisoformat(old_day)).days + 5, 30), 800)
        except ValueError:
            return None
        out = h.call("market.daily", code=d.ticker, as_of=today, lookback_days=lookback)
        new = fact_value(metric_facts(out.cards), "close")
        if new is None:
            return None
        # 除权除息后前复权序列会整体重算：用今日序列里档案交易日的复权价做基准，而不是档案里存的历史基准
        base = _rebase_close(self.deps.repo, out.cards[0], old_day)
        if base is None or base <= 0:
            return None                          # 无法对齐复权基准：宁可本次不报，也不制造假异动
        pct = float((new / base - 1) * 100)
        if abs(pct) < threshold:
            return None
        claims = [AffectedClaim(claim_id=c.id, text=c.text, reason=f"依赖档案时点的行情/估值，价格已变化 {pct:+.1f}%")
                  for c in d.claims if c.status == "verified"
                  and any((e := evidence.get(i)) and e.skill in PRICE_SOURCES for i in c.evidence_ids)]
        event = WatchEvent(kind="price_move", title=f"价格较档案时点变化 {pct:+.1f}%",
                           date=str(out.cards[0].metrics[0].period)[:10],
                           detail=f"{base} → {new}（同一复权基准，阈值 ±{threshold:g}%）", evidence_ids=[out.cards[0].id])
        return event, claims, out.cards

    @staticmethod
    def _triggers(d: Dossier, fired: set[str]) -> list[WatchEvent]:
        """档案自己写的复核条件：按关键词与本次变化类型粗匹配，只标「可能触发」，由复核确认。"""
        out = []
        for t in d.review_triggers:
            kinds = [k for k in fired if any(w in t.condition for w in TRIGGER_WORDS[k])]
            if kinds:
                out.append(WatchEvent(kind="trigger", title=t.condition,
                                      detail=f"可能因{'、'.join(EVENT_LABEL[k] for k in kinds)}而触发，复核时确认；{t.reason}"))
        return out

    # ---------- 增量复核 ----------
    def start_review(self, alert_id: str, service, background: bool = True) -> str:
        alert = self.store.get_alert(alert_id)
        if alert is None:
            raise WatchError("提醒不存在")
        if alert.status != "open":
            raise WatchError("只有待处理的提醒可以发起复核")
        instance = (getattr(service, "instance", None)
                    or getattr(getattr(service, "research", None), "instance", None))
        if not self.store.claim_review(alert_id, instance=instance):   # 原子领取：并发双击只有一个成功
            raise WatchError("提醒已被另一次复核操作接管")
        focus = [a.text for a in alert.affected][:8]
        changes = "；".join(e.title for e in alert.events)
        question = f"{alert.name}（{alert.ticker}）上次研究后有新变化：{changes}。请用新数据重新研究，说明哪些旧结论仍成立、哪些需要修正。"[:480]
        try:
            task_id = service.start(ResearchRequest(question=question, ticker=alert.ticker, route="standard"),
                                    background=background, review_focus=focus)
        except Exception:
            self.store.release_review(alert_id)             # 起任务失败：放回待处理，允许重试
            raise
        self.store.finish_review(alert_id, task_id)         # 回填真实任务号（sync 看到 pending 占位不会重置）
        return task_id

    def dismiss(self, alert_id: str, note: str) -> None:
        alert = self.store.get_alert(alert_id)
        if alert is None:
            raise WatchError("提醒不存在")
        if alert.status != "open":              # 复核中/已完成的提醒不能被忽略，否则新版本会与提醒失联
            raise WatchError("只有待处理的提醒可以忽略")
        self.store.update_alert(alert_id, status="dismissed", note=note)

    def sync(self) -> None:
        """复核任务结束后回写提醒状态；进程崩溃遗留的僵尸任务先标记失败，提醒随之解锁。"""
        self.deps.repo.recover_stale_tasks(RECOVER_AGE_S)
        for a in self.store.list_alerts(status="reviewing"):
            if a.review_task_id == "pending":   # 领取后正在创建任务：活租约不论多久都等，死租约给 120s 宽限
                if a.review_instance and self.deps.repo.lease_alive(a.review_instance):
                    continue                    # CLI 同步复核可能远超 120s：进程还在工作就不能重置
                try:
                    started = datetime.fromisoformat(a.review_started_at or "")
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                except ValueError:
                    started = None
                if started and (datetime.now(timezone.utc) - started).total_seconds() < 120:
                    continue
            task = self.deps.repo.get_task(a.review_task_id or "") if a.review_task_id != "pending" else None
            if not task:                        # 任务缺失/编号没写上：放回待处理，避免永久卡死
                self.store.update_alert(a.id, status="open", review_task_id=None,
                                        note="复核任务缺失或已清理，请重新发起复核")
                continue
            if task["status"] == "done":
                self.store.update_alert(a.id, status="resolved", resolved_version=(task.get("result") or {}).get("version"))
            elif task["status"] in ("failed", "cancelled"):
                self.store.update_alert(a.id, status="open", review_task_id=None, note=f"复核任务{task['status']}，可重试")


def alert_summary(alert: Alert, created: bool | None = None) -> str:
    """给人看的一句话：哪几类变化、影响几条旧结论。"""
    kinds = "、".join(dict.fromkeys(EVENT_LABEL[e.kind] for e in alert.events))
    head = {True: "有新变化", False: "这批变化已提醒过"}.get(created, "发现变化")
    return f"{head}（{kinds}），影响 {len(alert.affected)} 条旧结论，见下方提醒"


def _fingerprint(d: Dossier, events: list[WatchEvent]) -> str:
    """同一批变化只提醒一次：价格异动按「档案版本+异动」计，不随每日 bar/pct 变化。"""
    parts = [d.key, str(d.version)]
    for e in events:
        parts.append("price_move" if e.kind == "price_move" else f"{e.kind}|{e.title}|{e.date}")
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()[:24]
