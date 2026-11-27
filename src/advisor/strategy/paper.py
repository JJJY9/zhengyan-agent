"""前向模拟盘：按策略卡规则逐日向前推进，只用当天以前可得的档案与行情。

- 只能向前：日期必须晚于上一个净值点，且不早于模拟开始日——没有回填，也就没有「用未来考过去」；
- 每笔模拟成交都记下依据的档案版本与证据；
- 等权、整手（100 股）、不加杠杆；系统不执行任何真实交易。
"""
from datetime import date
import threading
from typing import Any, Callable

from pymongo.errors import DuplicateKeyError

from advisor.contracts import Dossier
from advisor.evidence import metric_facts
from advisor.graph.deps import Deps
from advisor.harness import Budget, BudgetExceeded, HarnessDenied, SkillFailed
from advisor.strategy.models import NavPoint, PaperTrade, Position, StrategyVersion
from advisor.strategy.store import StrategyStore

LOT = 100


class PaperError(ValueError):
    pass


class PaperBook:
    def __init__(self, deps: Deps, store: StrategyStore, open_alerts: Callable[[str], int] = lambda t: 0):
        self.deps = deps
        self.store = store
        self.open_alerts = open_alerts
        self.initial_cash = float(deps.policy.strategy.get("initial_cash", 1_000_000))
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, card_id: str, version: int) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(f"{card_id}:{version}", threading.Lock())

    # ---------- 信号：当天可见的事实 ----------
    def dossier_at(self, ticker: str, day: date) -> Dossier | None:
        """当天之前（含）研究时点最新的档案版本。

        版本号与 as_of 不一定同序：CLI --as-of 回填会产出「版本号更高但 as_of 更早」的档案，
        因此按 as_of（同日再按版本号）取最新，而不是按版本号取第一个。
        """
        candidates = [v for v in self.deps.repo.list_dossier_versions(ticker)
                      if str(v["as_of"])[:10] <= day.isoformat()]
        if not candidates:
            return None
        latest = max(candidates, key=lambda v: (str(v["as_of"])[:10], v["version"]))
        return self.deps.repo.get_dossier(ticker, latest["version"])

    def facts(self, h, ticker: str, day: date) -> tuple[dict[str, Any], Dossier | None, list[str]]:
        d = self.dossier_at(ticker, day)
        if d is None:
            return {}, None, []
        cards = self.deps.repo.get_evidence(d.evidence_ids)
        facts: dict[str, Any] = {k: m.value for k, m in metric_facts(cards).items()}
        facts.pop("close", None)          # 价格只认当天行情：档案里的是研究时点价，不是今天能成交的价
        ev_ids = [c.id for c in cards if c.source_type in ("market", "calc")]
        try:
            live = h.call("market.daily", code=ticker, as_of=day).cards
            live_facts = metric_facts(live)
            close = live_facts.get("close")
            if close is not None and str(close.period)[:10] == day.isoformat():
                facts.update({k: m.value for k, m in live_facts.items()})   # 行情用当天的，基本面用档案的
                ev_ids = [c.id for c in live] + ev_ids
                for c in live:
                    self.deps.repo.put_evidence(c)
            # 停牌/周末/节假日时数据源返回最后一条历史 K 线（period < day）：没有当日报价就不交易
        except (SkillFailed, HarnessDenied, BudgetExceeded):
            pass
        facts.update(status=d.status, verified=sum(c.status == "verified" for c in d.claims),
                     disputed=sum(c.status == "disputed" for c in d.claims), open_alerts=self.open_alerts(ticker))
        return facts, d, ev_ids[:6]

    def universe(self, card: StrategyVersion) -> list[str]:
        keys = card.universe or [d["key"] for d in self.deps.repo.list_dossiers()]
        return [k for k in keys if ":" not in k]   # 历史脏数据容错：评测命名空间 key 不参与取档

    # ---------- 推进一天 ----------
    def step(self, card: StrategyVersion, day: date) -> NavPoint:
        # 同一卡同一版本串行推进：调度器与手动补推并发不同日期时，不能基于同一快照各自成交
        with self._lock(card.card_id, card.version):
            return self._step_once(card, day)

    def _step_once(self, card: StrategyVersion, day: date) -> NavPoint:
        if card.status not in ("paper", "review", "promoted"):
            raise PaperError(f"{card.ref} 不在模拟中")
        if day > date.today():
            raise PaperError(f"{day} 晚于今天：模拟盘不能推进到未来")
        if card.paper_start and day < card.paper_start:
            raise PaperError(f"{day} 早于模拟开始日 {card.paper_start}：模拟盘只能向前")
        for _ in range(3):
            navs = self.store.navs_of(card.card_id, card.version)
            if navs and day <= navs[-1].date:
                if day == navs[-1].date:
                    return navs[-1]                                          # 同一天重复推进：幂等
                raise PaperError(f"{day} 不晚于上一个净值点 {navs[-1].date}：模拟盘只能向前")
            point, trades = self._compute(card, day, navs)
            # 序号即并发认领：取 max(seq)+1 而不是 len+1——删除补偿会留下序号空洞，len+1 会永远撞已存在的 seq
            point = point.model_copy(update={"seq": max((n.seq for n in navs), default=0) + 1})
            try:
                self.store.add_nav(point)
            except DuplicateKeyError:
                latest = self.store.navs_of(card.card_id, card.version)
                if latest and day <= latest[-1].date:
                    if day == latest[-1].date:
                        return latest[-1]
                    raise PaperError(f"{day} 不晚于上一个净值点 {latest[-1].date}：模拟盘只能向前")
                continue                       # 另一进程先推进了更早的日期：基于新净值重算
            try:
                self.store.add_trades(trades)
            except Exception:
                latest = self.store.navs_of(card.card_id, card.version)
                if latest and latest[-1].date != day:
                    # 已有更晚净值点基于今天成交推进：此时删除会让它们悬空，保留并显式要求人工核对
                    raise PaperError(f"{day} 成交写入失败且已有更晚净值点 {latest[-1].date}：该日成交缺失，需人工核对") from None
                self.store.delete_nav(card.card_id, card.version, day)   # 成交没写上：撤销净值点，重试补齐
                raise
            return point
        raise PaperError("并发推进冲突：请稍后重试")

    def _compute(self, card: StrategyVersion, day: date, navs: list[NavPoint]) -> tuple[NavPoint, list[PaperTrade]]:
        """按已有净值快照计算当天净值与成交（不做写入；并发认领由 _step_once 负责）。"""
        cash = navs[-1].cash if navs else self.initial_cash
        held = {p.ticker: p for p in (navs[-1].positions if navs else [])}
        h = self.deps.harness(f"paper_{card.card_id}_v{card.version}", "quick", day,
                              Budget(limits=self.deps.policy.route("quick").budget), node="paper", principal="paper")
        trades, notes = [], []
        snap = {t: self.facts(h, t, day) for t in dict.fromkeys(list(held) + self.universe(card))}
        # 有档案却没有任何标的当天有行情：周末/节假日/全部停牌。不生成净值点，
        # 否则晋级门控会把日历日当成「前向交易日」计数。
        if any(d is not None for _, d, _ in snap.values()) and not any(f.get("close") is not None for f, _, _ in snap.values()):
            raise PaperError(f"{day} 没有当日行情（非交易日或全部停牌），不生成净值点")

        def trade(t: str, side: str, shares: int, price: float, reason: str):
            _, d, ev = snap[t]
            trades.append(PaperTrade(card_id=card.card_id, version=card.version, date=day, ticker=t, side=side,
                                     shares=shares, price=price, reason=reason, dossier_version=d.version if d else None,
                                     dossier_as_of=d.as_of if d else None, evidence_ids=ev))

        # 1. 移出：任一退出规则满足、触发止损、或档案不可用（当天移出的不再当天纳入）
        closed_today: set[str] = set()
        for t, pos in list(held.items()):
            facts, d, _ = snap[t]
            price = float(facts["close"]) if facts.get("close") is not None else None
            if price is None:
                notes.append(f"{t} 今日无行情，持仓按上次价格计")
                continue
            pos.price = price
            hit = [r.label() for r in card.exit if r.check(facts)]
            if card.stop_loss_pct and price <= pos.cost * (1 - card.stop_loss_pct / 100):
                hit.append(f"止损 -{card.stop_loss_pct:g}%")
            if d is None:
                hit.append("档案不可用")
            if hit:
                cash += pos.shares * price
                trade(t, "close", pos.shares, price, "移出：" + "；".join(hit))
                del held[t]
                closed_today.add(t)

        # 2. 纳入：全部进入规则满足，等权、整手
        nav_now = cash + sum(p.shares * p.price for p in held.values())
        slot = nav_now / card.max_positions
        for t in self.universe(card):
            if len(held) >= card.max_positions or t in held or t in closed_today:
                continue
            facts, d, _ = snap[t]
            results = [(r, r.check(facts)) for r in card.entry]
            missing = [r.label() for r, ok in results if ok is None]
            if d is None or facts.get("close") is None or not all(ok for _, ok in results):
                if missing and d is not None:
                    notes.append(f"{t} 缺数据：{'、'.join(missing)}")
                continue
            price = float(facts["close"])
            shares = int(min(slot, cash) // (price * LOT)) * LOT
            if shares <= 0:
                notes.append(f"{t} 满足条件但资金不足一手")
                continue
            cash -= shares * price
            held[t] = Position(ticker=t, shares=shares, cost=price, price=price, opened=day)
            trade(t, "open", shares, price, "纳入：" + "；".join(r.label() for r in card.entry))

        point = NavPoint(card_id=card.card_id, version=card.version, date=day, cash=round(cash, 2),
                         nav=round(cash + sum(p.shares * p.price for p in held.values()), 2),
                         positions=list(held.values()), notes=notes)
        return point, trades


def nav_stats(navs: list[NavPoint], initial: float) -> dict:
    """累计收益与最大回撤（%）。"""
    if not navs:
        return {"days": 0, "return_pct": 0.0, "max_drawdown_pct": 0.0}
    peak, mdd = initial, 0.0
    for p in navs:
        peak = max(peak, p.nav)
        mdd = max(mdd, (peak - p.nav) / peak * 100 if peak else 0)
    return {"days": len(navs), "return_pct": round((navs[-1].nav / initial - 1) * 100, 2), "max_drawdown_pct": round(mdd, 2)}
