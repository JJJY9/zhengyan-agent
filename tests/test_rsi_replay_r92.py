"""R92a 回归：缓存副本新鲜度投影、冻结样本优先与增量补种、样本瘦身、
批准上限按预算推导与放弃原因、专家命名空间兜底区分文档缺失。"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from advisor.evals.compare import run_compare
from advisor.experts.agent import _payload_namespace
from advisor.rsi.evaluator import _s3_slices
from advisor.rsi.issues import cached_results
from advisor.rsi.ledger import Ledger
from tests.fakes import CODE, as_of

AS_OF = "2026-10-16"


def _h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _sid(q, count=8) -> str:
    return _h(f"ds:search:{AS_OF}:{count}:{q}")[:12]


def _raw(mark: str, authority=4):
    return [{"title": f"标题-{mark}", "url": f"https://www.stcn.com/{mark}.html", "authority": authority,
             "rerank_score": 2.0, "date": "2026-08-01T00:00:00+00:00", "summary": f"摘要-{mark}"}]


def _put(repo, key, raw, expires_hours=24):
    repo.db.cache.insert_one({"key": key, "value": raw,
                              "expires_at": datetime.now(timezone.utc) + timedelta(hours=expires_hours)})


def _deps(repo):
    return SimpleNamespace(repo=repo, policy=SimpleNamespace(search={}))


def test_freshest_copy_wins_with_projected_expiry(repo):
    """两份非空副本：按 expires_at 取更新的 legacy 副本，而不是键字典序（R90 的 rank 第二键曾失效）。"""
    q = "示例酒业 业绩 fresh"
    _put(repo, f"ds:search:user:{AS_OF}:8:{q}", _raw("old-95"), expires_hours=1)     # 旧 scoped 副本
    _put(repo, f"ds:search:{AS_OF}:8:{q}", _raw("new-96"), expires_hours=20)         # 新 legacy 副本
    urls = [c["url"] for c in cached_results(repo)[q]]
    assert urls == ["https://www.stcn.com/new-96.html"]


def test_frozen_samples_take_precedence_over_live_cache(repo):
    """同一 key 在 TTL 过期后被重搜改写：回放仍用冻结样本，保证同一套题。"""
    q = "示例酒业 业绩 frozen"
    key = f"ds:search:{AS_OF}:8:{q}"
    _put(repo, key, _raw("A"))
    slices, _ = _s3_slices(_deps(repo), Ledger(repo.db))                  # 冻结并持久化样本
    assert slices["replay"][0]["raw"][0]["url"].endswith("/A.html")
    repo.db.cache.update_one({"key": key}, {"$set": {"value": _raw("B", authority=1)}})   # 重搜改写了缓存
    slices2, _ = _s3_slices(_deps(repo), Ledger(repo.db))
    assert slices2["replay"][0]["raw"][0]["url"].endswith("/A.html")      # 仍按冻结样本回放
    doc = repo.db.rsi_protocols.find_one({"scenario": "S3"})
    sample = doc["replay_samples"][_sid(q)]
    assert set(sample["raw"][0]) == {"url", "authority", "rerank_score", "date"}   # 样本瘦身
    assert "title" not in sample["raw"][0]


def test_backfill_is_incremental_and_idempotent(repo):
    """旧协议补种不能一次性：先补到的子集之后仍要按缺失继续补齐，TTL 后不丢题。"""
    qa, qb = "示例酒业 业绩 A", "示例酒业 业绩 B"
    ids = [_sid(qa), _sid(qb)]
    repo.db.rsi_protocols.insert_one({"scenario": "S3", "replay_ids": ids, "hash": _h(ids),
                                      "created_at": datetime.now(timezone.utc)})
    _put(repo, f"ds:search:{AS_OF}:8:{qa}", _raw("A"))
    _s3_slices(_deps(repo), Ledger(repo.db))                              # 首评只有 A：补到部分
    doc = repo.db.rsi_protocols.find_one({"scenario": "S3"})
    assert len(doc["replay_samples"]) == 1
    _put(repo, f"ds:search:{AS_OF}:8:{qb}", _raw("B"))
    slices, _ = _s3_slices(_deps(repo), Ledger(repo.db))                  # B 出现：增量补齐
    assert len(slices["replay"]) == 2
    doc = repo.db.rsi_protocols.find_one({"scenario": "S3"})
    assert len(doc["replay_samples"]) == 2
    repo.db.cache.delete_many({})                                         # TTL 全过期
    slices3, _ = _s3_slices(_deps(repo), Ledger(repo.db))
    assert len(slices3["replay"]) == 2


class _StuckSvc:
    """评审永远停留在 waiting_approval：验证放弃分支与原因可见。"""

    def __init__(self, repo):
        self.deps = SimpleNamespace(repo=repo)
        self.resume_calls = 0
        self.research = SimpleNamespace(resume=self.resume)

    def start(self, req, background=False, namespace=None, **kw):
        self.deps.repo.create_task("t_stuck", req.model_dump(mode="json") | {"namespace": namespace})
        self.deps.repo.update_task("t_stuck", status="waiting_approval", interrupt={"kind": "approval"})
        return "t_stuck"

    def resume(self, tid, approve, background=False, **kw):
        self.resume_calls += 1
        self.deps.repo.update_task(tid, status="waiting_approval", interrupt={"kind": "approval"})


def test_compare_gives_up_with_reason(repo):
    svc = _StuckSvc(repo)
    run_compare(svc, [{"id": "c1", "question": f"复核 {CODE} 半年报业绩", "ticker": CODE, "as_of": str(as_of())}],
                ["standard"], run_id="ev_stuck")
    assert svc.resume_calls == 64                                        # 无 policy 桩：保守上限
    run = repo.get_eval("ev_stuck")
    entry = run["results"][0]
    assert "超过上限" in entry["error"]                                   # 放弃原因可见，不是 waiting_approval: None
    assert repo.get_task("t_stuck")["status"] == "cancelled"


def test_payload_namespace_distinguishes_missing_doc(repo):
    deps = SimpleNamespace(repo=repo)
    assert _payload_namespace(deps, {"task_id": "t_x", "namespace": "eval-r-standard"}) == "eval-r-standard"
    repo.create_task("t_user", {"question": "q"})                        # 用户任务：request 无 namespace
    assert _payload_namespace(deps, {"task_id": "t_user"}) == ""          # 不能误关进 unknown-* 隔离域
    repo.create_task("t_eval", {"question": "q", "namespace": "eval-r-deep"})
    assert _payload_namespace(deps, {"task_id": "t_eval"}) == "eval-r-deep"
    assert _payload_namespace(deps, {"task_id": "t_gone"}).startswith("unknown-")   # 文档缺失才 fail-closed
