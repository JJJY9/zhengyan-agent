"""R90a 回归：S3 冻结回放签名兼容与样本持久化、缓存身份去重、compare 多次自动批准。"""
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from advisor.evals.compare import run_compare
from advisor.rsi.evaluator import _cached_searches, _s3_slices
from advisor.rsi.issues import cached_results
from advisor.rsi.ledger import Ledger
from tests.fakes import CODE, as_of

AS_OF = "2026-10-16"
RAW = [{"title": f"t{i}", "url": f"https://www.stcn.com/a/{i}.html", "authority": 4,
        "rerank_score": 2.0, "date": "2026-08-01T00:00:00+00:00"} for i in range(8)]


def _h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _put(repo, key, raw=RAW):
    repo.db.cache.insert_one({"key": key, "value": raw,
                              "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc)})


def _freeze(ledger, ids):
    ledger.db.rsi_protocols.insert_one({"scenario": "S3", "replay_ids": sorted(ids), "hash": _h(sorted(ids)),
                                        "created_at": datetime.now(timezone.utc)})


def test_scoped_signature_from_95_window_loads_via_alias(repo):
    """#95 窗口按 scoped 全键哈希冻结的协议：升级后要用别名匹配回来，不能永久 0 加载。"""
    q = "示例酒业 业绩 alias"
    key = f"ds:search:user:{AS_OF}:8:{q}"
    _put(repo, key)
    # 模拟 #95 窗口冻结：id = _h(scoped 全键)[:12]
    scoped_id = _h(key)[:12]
    _freeze(Ledger(repo.db), [scoped_id])
    deps = SimpleNamespace(repo=repo, policy=SimpleNamespace(search={}))
    slices, replay_ids = _s3_slices(deps, Ledger(repo.db))
    assert scoped_id in replay_ids
    assert len(slices["replay"]) == 1                         # 别名匹配到样本
    # 样本已在本次调用中持久化：缓存被清掉后仍能按同一套题加载
    repo.db.cache.delete_many({})
    slices2, _ = _s3_slices(deps, Ledger(repo.db))
    assert len(slices2["replay"]) == 1


def test_legacy_freeze_persists_samples_survive_ttl(repo):
    """新冻结一并持久化样本：24h 缓存过期后协议不再永久失败。"""
    q = "示例酒业 业绩 ttl"
    _put(repo, f"ds:search:{AS_OF}:8:{q}")
    deps = SimpleNamespace(repo=repo, policy=SimpleNamespace(search={}))
    slices, _ = _s3_slices(deps, Ledger(repo.db))             # 首次评测冻结并持久化样本
    assert len(slices["replay"]) == 1
    doc = repo.db.rsi_protocols.find_one({"scenario": "S3"})
    assert doc["replay_samples"]
    repo.db.cache.delete_many({})                             # 模拟 TTL 过期
    slices2, _ = _s3_slices(deps, Ledger(repo.db))
    assert len(slices2["replay"]) == 1                        # 从冻结样本加载
    assert [(r["as_of"], r["query"]) for r in slices2["replay"]] == [(AS_OF, q)]


def test_empty_copy_does_not_swallow_nonempty(repo):
    """同一样本旧键为空、scoped 键有数据：两种插入顺序下都保留有数据副本。"""
    q = "示例酒业 业绩 empty"
    legacy = f"ds:search:{AS_OF}:8:{q}"
    scoped = f"ds:search:user:{AS_OF}:8:{q}"
    _put(repo, legacy, [])
    _put(repo, scoped, RAW)
    assert _cached_searches(repo)[0]["raw"] == RAW
    repo.db.cache.delete_many({})
    _put(repo, scoped, RAW)                                   # 反序插入
    _put(repo, legacy, [])
    assert _cached_searches(repo)[0]["raw"] == RAW


def test_content_drift_still_deduped_by_identity(repo):
    """同一次搜索的旧/新副本内容漂移（标题/顺序不同）时按逻辑身份折叠，阈值不被放大。"""
    q = "示例酒业 业绩 drift"
    drifted = [dict(r, title=r["title"] + "（重搜）") for r in RAW]
    _put(repo, f"ds:search:{AS_OF}:8:{q}", RAW)
    _put(repo, f"ds:search:user:{AS_OF}:8:{q}", drifted)
    assert len(cached_results(repo)[q]) == 8                  # 不是 16
    assert len(_cached_searches(repo)) == 1
    # 不同 count 是同一次搜索的说法不成立：仍各自计数
    _put(repo, f"ds:search:{AS_OF}:10:{q}", RAW)
    assert len(_cached_searches(repo)) == 2


class _LoopSvc:
    """最小服务桩：每次 resume 都重新进入待审批，第三次才结束（模拟多专家并行确认）。"""

    def __init__(self, repo):
        self.deps = SimpleNamespace(repo=repo)
        self.resume_calls = 0
        self.research = SimpleNamespace(resume=self.resume)

    def start(self, req, background=False, namespace=None, **kw):
        self.deps.repo.create_task("t_loop", req.model_dump(mode="json") | {"namespace": namespace})
        self.deps.repo.update_task("t_loop", status="waiting_approval", interrupt={"kind": "approval"})
        return "t_loop"

    def resume(self, tid, approve, background=False, **kw):
        self.resume_calls += 1
        if self.resume_calls < 3:
            self.deps.repo.update_task(tid, status="waiting_approval", interrupt={"kind": "approval"})
        else:
            self.deps.repo.update_task(tid, status="failed", error="模拟结束")


def test_compare_approves_repeated_interrupts(repo, policy):
    """confirm 审批/多专家并行会多次进入待审批：评测内自动批准必须循环，不能只批一次后悬挂。"""
    svc = _LoopSvc(repo)
    run_compare(svc, [{"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())}],
                ["standard"], run_id="ev_loop")
    assert svc.resume_calls == 3
    run = repo.get_eval("ev_loop")
    assert run["results"][0]["auto_approved"] is True
    assert "waiting_approval" not in (run["results"][0].get("error") or "")
