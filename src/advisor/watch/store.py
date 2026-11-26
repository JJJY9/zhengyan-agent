"""订阅与提醒的存储。提醒按指纹去重：同一批变化只提醒一次。"""
from datetime import datetime, timezone

from pymongo import DESCENDING
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from advisor.watch.models import Alert, Subscription


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WatchStore:
    def __init__(self, db: Database):
        self.subs = db.watch_subscriptions
        self.alerts = db.watch_alerts
        self.subs.create_index("ticker", unique=True)
        self.alerts.create_index("id", unique=True)
        self.alerts.create_index("fingerprint", unique=True)

    # ---------- 订阅 ----------
    def subscribe(self, ticker: str, name: str = "", price_move_pct: float = 10) -> Subscription:
        existing = self.get_sub(ticker)
        sub = Subscription(ticker=ticker, name=name or (existing.name if existing else ""), price_move_pct=price_move_pct,
                           created_at=existing.created_at if existing else _now(),
                           last_scan_at=existing.last_scan_at if existing else None,
                           last_scan_note=existing.last_scan_note if existing else "")
        self.subs.update_one({"ticker": ticker}, {"$set": sub.model_dump(mode="json")}, upsert=True)
        return sub

    def unsubscribe(self, ticker: str) -> None:
        self.subs.update_one({"ticker": ticker}, {"$set": {"active": False}})

    def get_sub(self, ticker: str) -> Subscription | None:
        doc = self.subs.find_one({"ticker": ticker}, projection={"_id": 0})
        return Subscription.model_validate(doc) if doc else None

    def list_subs(self, active_only: bool = True) -> list[Subscription]:
        q = {"active": True} if active_only else {}
        return [Subscription.model_validate(d) for d in self.subs.find(q, projection={"_id": 0}).sort("ticker", 1)]

    def mark_scanned(self, ticker: str, note: str) -> None:
        self.subs.update_one({"ticker": ticker}, {"$set": {"last_scan_at": _now().isoformat(), "last_scan_note": note}})

    # ---------- 提醒 ----------
    def add_alert(self, alert: Alert) -> tuple[Alert, bool]:
        """返回 (提醒, 是否新建)。指纹相同说明这批变化已经提醒过。"""
        try:
            self.alerts.insert_one(alert.model_dump(mode="json"))
            return alert, True
        except DuplicateKeyError:
            doc = self.alerts.find_one({"fingerprint": alert.fingerprint}, projection={"_id": 0})
            return Alert.model_validate(doc), False

    def get_alert(self, alert_id: str) -> Alert | None:
        doc = self.alerts.find_one({"id": alert_id}, projection={"_id": 0})
        return Alert.model_validate(doc) if doc else None

    def claim_review(self, alert_id: str, instance: str | None = None) -> bool:
        """原子领取复核：状态、占位 task_id、领取时间与领取方租约同一次写入。

        分两步写（先置 reviewing 再回填 task_id）时，并发 sync 会把中间态当成「任务缺失」重置回 open，
        复核完成后永远无法 resolve；占位 pending + 领取时间让 sync 能区分「正在创建」与「真的丢了」，
        领取方租约则让跨进程同步复核（可能远超 120s）不被误判。"""
        fields = {"status": "reviewing", "review_task_id": "pending",
                  "review_started_at": _now().isoformat()}
        if instance:
            fields["review_instance"] = instance
        doc = self.alerts.find_one_and_update(
            {"id": alert_id, "status": "open"},
            {"$set": fields},
        )
        return doc is not None

    def finish_review(self, alert_id: str, task_id: str) -> None:
        res = self.alerts.update_one({"id": alert_id, "review_task_id": "pending"},
                                     {"$set": {"review_task_id": task_id},
                                      "$unset": {"review_started_at": "", "review_instance": ""}})
        if res.matched_count == 0:
            # 占位在任务创建期间被 sync 误判重置回 open：任务已真实创建，把提醒接回复核中
            # （已被忽略或被另一次领取的提醒不接回，避免把用户操作覆盖掉）
            self.alerts.update_one({"id": alert_id, "review_task_id": None, "status": "open"},
                                   {"$set": {"status": "reviewing", "review_task_id": task_id},
                                    "$unset": {"review_instance": ""}})

    def release_review(self, alert_id: str) -> None:
        """起任务失败：把领取放回待处理，允许重试。"""
        self.alerts.update_one({"id": alert_id, "review_task_id": "pending"},
                               {"$set": {"status": "open", "review_task_id": None},
                                "$unset": {"review_started_at": "", "review_instance": ""}})

    def list_alerts(self, ticker: str | None = None, status: str | None = None, limit: int = 100) -> list[Alert]:
        q = {k: v for k, v in (("ticker", ticker), ("status", status)) if v}
        cur = self.alerts.find(q, projection={"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        return [Alert.model_validate(d) for d in cur]

    def update_alert(self, alert_id: str, **fields) -> None:
        self.alerts.update_one({"id": alert_id}, {"$set": fields})
