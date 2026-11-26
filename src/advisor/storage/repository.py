"""MongoDB 存储：档案（版本链）、证据（只插入）、快照、审计、任务、缓存。"""
from datetime import date, datetime, timedelta, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from advisor.contracts import Dossier, EvidenceCard


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clean(doc: dict | None) -> dict | None:
    if doc is not None:
        doc.pop("_id", None)
    return doc


class Repository:
    def __init__(self, db: Database):
        self.db = db

    @classmethod
    def connect(cls, uri: str, db_name: str) -> "Repository":
        client = MongoClient(uri, serverSelectionTimeoutMS=3000, tz_aware=True)
        repo = cls(client[db_name])
        repo.ensure_indexes()
        return repo

    def ensure_indexes(self) -> None:
        self.db.evidence.create_index("id", unique=True)
        self.db.snapshots.create_index("hash", unique=True)
        self.db.dossiers.create_index([("key", ASCENDING), ("version", DESCENDING)], unique=True)
        self.db.tasks.create_index("task_id", unique=True)
        self.db.audit.create_index([("task_id", ASCENDING), ("ts", ASCENDING)])
        self.db.cache.create_index("key", unique=True)
        self.db.cache.create_index("expires_at", expireAfterSeconds=0)            # 过期缓存自动清理，防止无限增长
        self.db.working_memory.create_index([("task_id", ASCENDING), ("created_at", ASCENDING)])
        self.db.working_memory.create_index([("task_id", ASCENDING), ("kind", ASCENDING), ("key", ASCENDING)],
                                            unique=True)
        self.db.working_memory.create_index("expires_at", expireAfterSeconds=0)   # 研究记忆到期自动清理
        self.db.eval_runs.create_index("run_id", unique=True)
        self.db.leases.create_index("instance", unique=True)

    def ping(self) -> bool:
        self.db.command("ping")
        return True

    # ---------- 证据：插入后不可变 ----------
    def put_snapshot(self, digest: str, content: str) -> None:
        try:
            self.db.snapshots.insert_one({"hash": digest, "content": content, "created_at": _now()})
        except DuplicateKeyError:
            pass

    def get_snapshot(self, digest: str) -> str | None:
        doc = self.db.snapshots.find_one({"hash": digest})
        return doc["content"] if doc else None

    def put_evidence(self, card: EvidenceCard) -> bool:
        """插入证据；同 id 已存在时比对内容：完全一致返回 True，元数据不同返回 False 并保留旧版（证据不可变）。"""
        new = card.model_dump(mode="json")
        try:
            self.db.evidence.insert_one(new)
            return True
        except DuplicateKeyError:
            existing = self.db.evidence.find_one({"id": card.id}, projection={"_id": 0})
            if existing is None:
                return True
            # 检索会话元数据（retrieved_at/rerank_score/excerpt/caveats/cluster_id…）随查询与
            # 结果集变化，不算冲突；只有来源与内容决定的身份字段不同才算元数据冲突
            identity = ("content", "source_type", "origin_id", "as_of", "snapshot_hash",
                        "title", "published_at")
            immutable = {k: v for k, v in new.items() if k in identity}
            return not [k for k in immutable if existing.get(k) != immutable[k]]

    def get_evidence(self, ids: list[str]) -> list[EvidenceCard]:
        docs = self.db.evidence.find({"id": {"$in": ids}})
        return [EvidenceCard.model_validate(_clean(d)) for d in docs]

    # ---------- 档案：版本链，不覆盖 ----------
    def latest_dossier_version(self, key: str) -> int | None:
        doc = self.db.dossiers.find_one({"key": key}, sort=[("version", DESCENDING)], projection={"version": 1})
        return doc["version"] if doc else None

    def latest_dossier_version_at(self, key: str, as_of: date) -> int | None:
        """as_of 当天或之前研究时点最新的版本号（同日取版本号最大）。

        回填研究（CLI --as-of）会产出「版本号更高但 as_of 更早」的档案，按版本号取
        「上一版」会把未来档案注入提示词（前视污染），所以按 as_of 取。
        """
        rows = [v for v in self.list_dossier_versions(key) if str(v["as_of"])[:10] <= as_of.isoformat()]
        if not rows:
            return None
        return max(rows, key=lambda v: (str(v["as_of"])[:10], v["version"]))["version"]

    def current_dossier(self, key: str, as_of: date | None = None) -> Dossier | None:
        """「当前档案」按研究时点取（同日再按版本号），而不是版本号最大：
        回填研究产出的高版本旧时点档案不能被当成最新结论（watch、个人情境官等读取都用这个）。"""
        version = self.latest_dossier_version_at(key, as_of or date.today())
        return self.get_dossier(key, version) if version is not None else None

    def insert_dossier(self, dossier: Dossier) -> None:
        self.db.dossiers.insert_one(dossier.model_dump(mode="json"))

    def try_insert_dossier(self, dossier: Dossier) -> bool:
        """插入档案；同 key 并发提交撞版本号时返回 False（由调用方重算版本后重试）。"""
        try:
            self.db.dossiers.insert_one(dossier.model_dump(mode="json"))
            return True
        except DuplicateKeyError:
            return False

    def get_dossier(self, key: str, version: int | None = None) -> Dossier | None:
        query: dict[str, Any] = {"key": key}
        if version is not None:
            query["version"] = version
        doc = self.db.dossiers.find_one(query, sort=[("version", DESCENDING)])
        return Dossier.model_validate(_clean(doc)) if doc else None

    def list_dossier_versions(self, key: str) -> list[dict]:
        cur = self.db.dossiers.find(
            {"key": key},
            projection={"_id": 0, "key": 1, "version": 1, "status": 1, "as_of": 1, "question": 1, "created_at": 1, "task_id": 1},
        ).sort("version", DESCENDING)
        return list(cur)

    def list_dossiers(self) -> list[dict]:
        pipeline = [
            {"$match": {"key": {"$not": {"$regex": ":"}}}},     # 评测等命名空间档案不进用户列表
            {"$sort": {"as_of": -1, "version": -1}},          # 「当前档案」= (as_of, version) 最大，回填的高版本旧档案不算最新
            {"$group": {
                "_id": "$key",
                "key": {"$first": "$key"}, "ticker": {"$first": "$ticker"}, "name": {"$first": "$name"},
                "version": {"$first": "$version"}, "status": {"$first": "$status"},
                "as_of": {"$first": "$as_of"}, "summary": {"$first": "$summary"},
                "created_at": {"$first": "$created_at"},
            }},
            {"$sort": {"created_at": -1}},
            {"$project": {"_id": 0}},
        ]
        return list(self.db.dossiers.aggregate(pipeline))

    # ---------- 任务 ----------
    def create_task(self, task_id: str, request: dict, instance: str | None = None) -> None:
        self.db.tasks.insert_one({
            "task_id": task_id, "request": request, "status": "queued", "events": [], "instance": instance,
            "interrupt": None, "result": None, "error": None, "created_at": _now(), "updated_at": _now(),
        })

    # ---------- 进程租约：回收僵尸任务只信租约，不信队列任务的年龄 ----------
    def touch_lease(self, instance: str) -> None:
        self.db.leases.update_one({"instance": instance}, {"$set": {"ts": _now()}}, upsert=True)

    def lease_alive(self, instance: str, ttl_s: float = 180.0) -> bool:
        doc = self.db.leases.find_one({"instance": instance})
        if not doc or not doc.get("ts"):
            return False
        ts = doc["ts"] if doc["ts"].tzinfo else doc["ts"].replace(tzinfo=timezone.utc)
        return ts > _now() - timedelta(seconds=ttl_s)

    def mark_running(self, task_id: str, instance: str | None = None) -> bool:
        """CAS 启动：只有仍处于 queued/running 的任务才真正执行，并重绑归属。

        回收/取消可能已把任务标为 failed/cancelled：此时绝不能再「复活」执行。"""
        fields = {"status": "running", "updated_at": _now()}
        if instance:
            fields["instance"] = instance
        doc = self.db.tasks.find_one_and_update(
            {"task_id": task_id, "status": {"$in": ["queued", "running"]}},
            {"$set": fields},
        )
        return doc is not None

    def finish_task(self, task_id: str, status: str, instance: str | None = None,
                    repair_recovered: bool = False, **fields: Any) -> bool:
        """收尾：默认只允许从 status=running 写终态（被回收/取消的任务不复活）。

        租约可能被误判（启动回收按 90s、快 tick 按 180s 判过期；休眠/时钟跳变/Mongo 短断）：此时任务已被回收成
        failed，但执行者其实还活着且带着完整结果。instance+repair_recovered 给出修复通道：
        归属仍是本实例、失败原因恰为租约回收时，允许用真实结果修正终态，不丢弃成果。
        """
        payload = {**fields, "status": status, "updated_at": _now()}
        if status != "failed":
            payload.setdefault("error", None)      # waiting_approval/cancelled 不能残留「已被回收」红字
        if self.db.tasks.update_one({"task_id": task_id, "status": "running"}, {"$set": payload}).matched_count == 1:
            return True
        if instance and repair_recovered:
            doc = self.db.tasks.find_one_and_update(
                {"task_id": task_id, "instance": instance, "status": "failed",
                 "error": {"$regex": "所属进程已不在"}},
                {"$set": payload},
            )
            return doc is not None
        return False

    def cancel_queued_tasks(self, instance: str, reason: str) -> int:
        """服务停机：本实例还在排队、注定不会再执行的任务直接标取消。"""
        res = self.db.tasks.update_many(
            {"status": "queued", "instance": instance},
            {"$set": {"status": "cancelled", "error": reason, "updated_at": _now()}},
        )
        return res.modified_count

    def update_task(self, task_id: str, **fields: Any) -> None:
        fields["updated_at"] = _now()
        self.db.tasks.update_one({"task_id": task_id}, {"$set": fields})

    def try_resume_task(self, task_id: str, instance: str | None = None) -> bool:
        """原子地把「待审批」任务切到运行中；并发双击/客户端重试只有一个能成功。

        恢复方通常与创建方不是同一进程：同一 CAS 里把 instance 重绑到当前进程，
        否则租约回收会把刚被批准的任务当成旧进程遗留标失败（审批静默丢失）。"""
        fields = {"status": "running", "interrupt": None, "error": None, "updated_at": _now()}
        if instance:
            fields["instance"] = instance
        doc = self.db.tasks.find_one_and_update(
            {"task_id": task_id, "status": "waiting_approval"},
            {"$set": fields},
        )
        return doc is not None

    # ---------- 任务级运行时账本（与图状态解耦：节点被审批中断/重放也不丢账） ----------
    def task_charge(self, task_id: str, kind: str, amount: float) -> None:
        """每次真实消耗原子累加；任务文档不存在的调用方（paper/watch 等虚拟任务）是无害空操作。"""
        if amount:
            self.db.tasks.update_one({"task_id": task_id}, {"$inc": {f"runtime_budget.{kind}": amount}})

    def task_runtime_budget(self, task_id: str) -> dict:
        doc = self.db.tasks.find_one({"task_id": task_id}, projection={"runtime_budget": 1})
        return dict((doc or {}).get("runtime_budget") or {})

    def add_paused_time(self, task_id: str, pause_s: float) -> None:
        """审批/停机等待累计：预算检查按开始时间减暂停时长计算已用执行时间。"""
        if pause_s > 0:
            self.db.tasks.update_one({"task_id": task_id}, {"$inc": {"paused_s": pause_s}})

    def task_paused_s(self, task_id: str) -> float:
        doc = self.db.tasks.find_one({"task_id": task_id}, projection={"paused_s": 1})
        return float((doc or {}).get("paused_s") or 0.0)

    def push_event(self, task_id: str, node: str, message: str) -> None:
        self.db.tasks.update_one(
            {"task_id": task_id},
            {"$push": {"events": {"ts": _now(), "node": node, "message": message}}, "$set": {"updated_at": _now()}},
        )

    def request_cancel(self, task_id: str) -> str | None:
        """撤销授权/取消（原子条件更新，绝不回写读到的旧状态）。

        等待确认的任务立即取消；运行中的任务只置取消标志，在下一个检查点停止。
        旧实现先读状态再无条件写回：任务在读写间隙完成时会被改回 running 且无人推进。
        """
        doc = self.db.tasks.find_one_and_update(
            {"task_id": task_id, "status": "waiting_approval"},
            {"$set": {"status": "cancelled", "cancel_requested": True, "updated_at": _now()}},
        )
        if doc is not None:
            return "cancelled"
        doc = self.db.tasks.find_one_and_update(
            {"task_id": task_id, "status": {"$in": ["queued", "running"]}},
            {"$set": {"cancel_requested": True, "updated_at": _now()}},
        )
        return doc["status"] if doc else None

    def is_cancel_requested(self, task_id: str) -> bool:
        doc = self.db.tasks.find_one({"task_id": task_id}, projection={"cancel_requested": 1})
        return bool(doc and doc.get("cancel_requested"))

    def recover_stale_tasks(self, max_age_s: float, lease_ttl_s: float = 180.0,
                            exclude_instance: str | None = None) -> int:
        """回收无主任务：所属进程租约失效的 queued/running 一律标记失败。

        排队中的任务不写 updated_at，只看年龄会误杀活进程的排队任务（实测：排队>30 分钟后被
        标 failed、随后线程池照常跑成 done）。以实例租约为准；历史数据（无 instance）退回按
        停滞年龄判断。exclude_instance 给回收方自己的实例：回收前先续自己的租约并跳过自己名下
        的任务——快 tick 与心跳竞争时，活进程绝不能回收自己正在执行的任务。
        """
        if exclude_instance:
            self.touch_lease(exclude_instance)
        cutoff = _now() - timedelta(seconds=max_age_s)
        ids: list[str] = []
        alive: dict[str, bool] = {}
        for d in self.db.tasks.find({"status": {"$in": ["queued", "running"]}},
                                    projection={"task_id": 1, "instance": 1, "updated_at": 1}):
            inst = d.get("instance")
            if inst:
                if exclude_instance and inst == exclude_instance:
                    continue                  # 自己的任务由自己的收尾负责
                if inst not in alive:
                    alive[inst] = self.lease_alive(inst, lease_ttl_s)
                if alive[inst]:
                    continue                      # 进程还活着（哪怕任务在排队一动不动）
            elif not (d.get("updated_at") and d["updated_at"] < cutoff):
                continue                          # 历史数据没有租约：退回按停滞年龄判断
            ids.append(d["task_id"])
        if not ids:
            return 0
        matched = [d["task_id"] for d in self.db.tasks.find(
            {"task_id": {"$in": ids}, "status": {"$in": ["queued", "running"]}}, projection={"task_id": 1})]
        if not matched:
            return 0
        self.db.tasks.update_many(
            {"task_id": {"$in": matched}, "status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "failed", "updated_at": _now(),
                      "error": "任务所属进程已不在（崩溃或无租约），已标记失败；可重新发起"}},
        )
        for task_id in matched:
            self.push_event(task_id, "runner", "所属进程已不在，已自动标记失败，可重新发起")
        return len(matched)

    def get_task(self, task_id: str) -> dict | None:
        return _clean(self.db.tasks.find_one({"task_id": task_id}))

    def list_tasks(self, limit: int = 30) -> list[dict]:
        """用户侧任务列表：评测等隔离命名空间的任务不出现（只在评测页可见）。"""
        cur = self.db.tasks.find({"request.namespace": None}, projection={"_id": 0, "events": 0}) \
            .sort("created_at", DESCENDING).limit(limit)
        return list(cur)

    # ---------- 工作记忆（研究记忆：计划、专家产出、质询；到期清理） ----------
    def save_working(self, task_id: str, kind: str, key: str, data: Any, ttl_days: float = 30) -> None:
        self.db.working_memory.update_one(
            {"task_id": task_id, "kind": kind, "key": key},
            {"$set": {"data": data, "created_at": _now(), "expires_at": _now() + timedelta(days=ttl_days)}},
            upsert=True,
        )

    def get_working(self, task_id: str, kind: str, key: str) -> Any | None:
        doc = self.db.working_memory.find_one({"task_id": task_id, "kind": kind, "key": key})
        return doc.get("data") if doc else None

    def list_working(self, task_id: str) -> list[dict]:
        cur = self.db.working_memory.find({"task_id": task_id}, projection={"_id": 0}).sort("created_at", ASCENDING)
        return list(cur)

    # ---------- 评测 ----------
    def save_eval(self, run: dict, only_if_running: bool = False) -> bool:
        """保存评测；only_if_running 用于作业写回前确认未被停机/回收标成 failed。"""
        if only_if_running:
            res = self.db.eval_runs.update_one({"run_id": run["run_id"], "status": "running"},
                                               {"$set": {**run, "updated_at": _now()}})
            return res.matched_count == 1
        self.db.eval_runs.update_one({"run_id": run["run_id"]}, {"$set": {**run, "updated_at": _now()}}, upsert=True)
        return True

    def list_evals(self, limit: int = 20) -> list[dict]:
        return list(self.db.eval_runs.find({}, projection={"_id": 0}).sort("created_at", DESCENDING).limit(limit))

    @staticmethod
    def _interrupted_verdict(done_n: int) -> dict:
        return {"decision": "insufficient", "provisional": True,
                "text": f"评测中断：已完成 {done_n} 例，结论不可用；可重新发起"}

    def _fail_eval_docs(self, docs: list[dict], reason: str) -> int:
        """把 running 评测逐条标 failed，并把临时结论改写为「中断」：失败行不能再显示「进行中」。

        summary 一律写完整形状（routes 至少为空映射）：点路径只写 verdict 会造出缺 routes 的残缺
        summary，前端汇总表按 s.routes 取键会整页崩溃。
        """
        n = 0
        for d in docs:
            summary = dict(d.get("summary") or {})
            summary.setdefault("routes", {})
            summary["verdict"] = self._interrupted_verdict(len(d.get("results") or []))
            n += self.db.eval_runs.update_one(
                {"run_id": d["run_id"], "status": "running"},
                {"$set": {"status": "failed", "updated_at": _now(), "error": reason,
                          "summary": summary}},
            ).modified_count
        return n

    def recover_stale_evals(self, max_age_s: float = 3600, lease_ttl_s: float = 180.0) -> int:
        """回收僵尸评测：所属进程租约已死的 running 评测立即标 failed；
        历史无 instance 的文档退回按停滞时长（进度每例刷新 updated_at）。"""
        cutoff = _now() - timedelta(seconds=max_age_s)
        stale: list[dict] = []
        alive: dict[str, bool] = {}
        for d in self.db.eval_runs.find({"status": "running"},
                                        projection={"run_id": 1, "instance": 1, "updated_at": 1,
                                                    "results": 1, "summary": 1}):
            inst = d.get("instance")
            stalled = bool(d.get("updated_at") and d["updated_at"] < cutoff)
            if inst:
                if inst not in alive:
                    alive[inst] = self.lease_alive(inst, lease_ttl_s)
                if alive[inst] and not stalled:
                    continue                      # 进程还活着且仍有进展
            elif not stalled:
                continue                          # 历史无 instance：退回按停滞时长
            stale.append(d)
        return self._fail_eval_docs(stale, "评测所属进程已不在，已标记失败；可重新发起")

    def fail_instance_evals(self, instance: str, reason: str) -> int:
        """优雅停机：本实例登记且未完成的评测直接标 failed（排队作业已被取消）。"""
        docs = list(self.db.eval_runs.find({"status": "running", "instance": instance},
                                           projection={"run_id": 1, "results": 1, "summary": 1}))
        return self._fail_eval_docs(docs, reason)

    def get_eval(self, run_id: str) -> dict | None:
        return _clean(self.db.eval_runs.find_one({"run_id": run_id}))

    # ---------- 审计 ----------
    def append_audit(self, record: dict) -> None:
        self.db.audit.insert_one({**record, "ts": _now()})

    def list_audit(self, task_id: str) -> list[dict]:
        return [_clean(d) for d in self.db.audit.find({"task_id": task_id}).sort("ts", ASCENDING)]

    # ---------- 缓存 ----------
    def cache_get(self, key: str) -> Any | None:
        doc = self.db.cache.find_one({"key": key})
        if not doc:
            return None
        expires = doc["expires_at"]
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return doc["value"] if expires > _now() else None

    def cache_set(self, key: str, value: Any, ttl_hours: float) -> None:
        self.db.cache.update_one(
            {"key": key},
            {"$set": {"value": value, "expires_at": _now() + timedelta(hours=ttl_hours)}},
            upsert=True,
        )
