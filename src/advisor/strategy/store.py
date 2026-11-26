"""策略卡、模拟成交与净值的存储。版本只追加；成交与净值只追加；状态变化写事件。"""
import hashlib
import json
import re
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from advisor.strategy.models import NavPoint, PaperTrade, StrategySpec, StrategyVersion


def _now() -> datetime:
    return datetime.now(timezone.utc)


def spec_hash(spec: StrategySpec) -> str:
    raw = json.dumps(StrategySpec.model_validate(spec.model_dump()).model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class StrategyStore:
    def __init__(self, db: Database):
        self.cards = db.strategy_cards
        self.trades = db.paper_trades
        self.navs = db.paper_navs
        self.events = db.strategy_events
        self.cards.create_index([("card_id", ASCENDING), ("version", ASCENDING)], unique=True)
        # 同一张卡只能有一个已晋级版本：并发 approve 时后到者会被唯一索引拦下
        self.cards.create_index("card_id", unique=True, partialFilterExpression={"status": "promoted"})
        self.navs.create_index([("card_id", ASCENDING), ("version", ASCENDING), ("date", ASCENDING)], unique=True)
        # 并发认领：不同进程基于同一快照推进不同日期时，(card, version, seq) 唯一让后写者失败并重算；
        # 历史数据没有 seq，用部分索引避开
        self.navs.create_index([("card_id", ASCENDING), ("version", ASCENDING), ("seq", ASCENDING)], unique=True,
                               partialFilterExpression={"seq": {"$gt": 0}})
        self.trades.create_index([("card_id", ASCENDING), ("version", ASCENDING), ("date", ASCENDING)])

    # ---------- 版本 ----------
    def next_card_id(self) -> str:
        nums = [int(m.group(1)) for cid in self.cards.distinct("card_id")
                if (m := re.match(r"^SC-(\d+)$", cid))]
        return f"SC-{max(nums, default=0) + 1:03d}"

    def insert(self, v: StrategyVersion) -> None:
        self.cards.insert_one(v.model_dump(mode="json"))

    def save(self, v: StrategyVersion, expect_status: str | None = None) -> bool:
        """只更新状态类字段；规则内容由 sha256 锁定，不能原地修改。

        expect_status 给定时以「状态未变」做乐观锁，返回是否写入成功（并发操作不会互相覆盖）。
        """
        fields = v.model_dump(mode="json", include={"status", "paper_start", "promoted_at", "gates", "approvals", "reject_reason"})
        q: dict = {"card_id": v.card_id, "version": v.version, "sha256": v.sha256}
        if expect_status is not None:
            q["status"] = expect_status
        res = self.cards.update_one(q, {"$set": fields})
        return res.matched_count == 1

    def get(self, card_id: str, version: int | None = None) -> StrategyVersion | None:
        q: dict = {"card_id": card_id} | ({"version": version} if version is not None else {})
        doc = self.cards.find_one(q, sort=[("version", DESCENDING)], projection={"_id": 0})
        return StrategyVersion.model_validate(doc) if doc else None

    def versions(self, card_id: str | None = None, status: list[str] | None = None) -> list[StrategyVersion]:
        q: dict = {}
        if card_id:
            q["card_id"] = card_id
        if status:
            q["status"] = {"$in": status}
        cur = self.cards.find(q, projection={"_id": 0}).sort([("card_id", ASCENDING), ("version", DESCENDING)])
        return [StrategyVersion.model_validate(d) for d in cur]

    # ---------- 模拟盘 ----------
    def add_trades(self, trades: list[PaperTrade]) -> None:
        if trades:
            self.trades.insert_many([t.model_dump(mode="json") for t in trades])

    def list_trades(self, card_id: str, version: int) -> list[PaperTrade]:
        cur = self.trades.find({"card_id": card_id, "version": version}, projection={"_id": 0}).sort("date", ASCENDING)
        return [PaperTrade.model_validate(d) for d in cur]

    def add_nav(self, p: NavPoint) -> None:
        self.navs.insert_one(p.model_dump(mode="json"))

    def delete_nav(self, card_id: str, version: int, day) -> None:
        """成交写入失败时撤销当天净值点，让重试能重新生成净值与成交。"""
        self.navs.delete_one({"card_id": card_id, "version": version, "date": day.isoformat()})

    def navs_of(self, card_id: str, version: int) -> list[NavPoint]:
        cur = self.navs.find({"card_id": card_id, "version": version}, projection={"_id": 0}).sort("date", ASCENDING)
        return [NavPoint.model_validate(d) for d in cur]

    # ---------- 事件 ----------
    def event(self, action: str, actor: str, card_id: str, version: int, detail: str, **extra) -> None:
        self.events.insert_one({"ts": _now(), "action": action, "actor": actor, "card_id": card_id, "version": version,
                                "detail": detail, **extra})

    def list_events(self, card_id: str | None = None, limit: int = 100) -> list[dict]:
        q = {"card_id": card_id} if card_id else {}
        return list(self.events.find(q, projection={"_id": 0}).sort("ts", DESCENDING).limit(limit))
