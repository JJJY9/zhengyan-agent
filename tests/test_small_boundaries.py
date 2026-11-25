"""小边界回归：审批等级拼写校验（fail closed）、令牌桶低速率/非法速率、登录会话表并发淘汰。"""
import threading

import pytest

from advisor.api.app import _SessionStore
from advisor.harness.policy import Policy
from advisor.harness.ratelimit import TokenBucket


# ---------- 审批等级配置校验 ----------

def test_unknown_approval_level_rejected():
    raw = {"routes": {}, "approvals": {"default": "auto", "skills": {"web.search": "Deny"}}}
    with pytest.raises(ValueError, match="审批等级"):
        Policy(raw, {})                                   # 拼写偏差会被当成 auto 放行：启动即拒绝加载
    Policy({"routes": {}, "approvals": {"default": "auto", "skills": {"web.search": "confirm"}}}, {})


# ---------- 令牌桶边界 ----------

def test_token_bucket_low_rate_does_not_hang():
    bucket = TokenBucket(0.5)                             # 容量至少 1：首个令牌必须立即可得
    t = threading.Thread(target=bucket.acquire, daemon=True)
    t.start()
    t.join(timeout=1.0)
    assert not t.is_alive()                               # 旧实现容量=rate<1：永远补不满一个令牌


def test_token_bucket_rejects_non_positive_rate():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            TokenBucket(bad)


# ---------- 登录会话表并发淘汰 ----------

class _SyncGetDict(dict):
    """让两个线程在「计算最早到期 token」时对齐：把并发窗口放大成确定时序。"""

    def __init__(self, barrier: threading.Barrier):
        super().__init__()
        self._barrier = barrier
        self._tls = threading.local()

    def get(self, key, default=None):
        v = super().get(key, default)
        if not getattr(self._tls, "hit", False):
            self._tls.hit = True
            try:
                self._barrier.wait(timeout=0.3)
            except threading.BrokenBarrierError:
                pass
        return v


def test_session_cap_holds_under_concurrent_create():
    store = _SessionStore(ttl_s=3600, max_sessions=3)
    for _ in range(3):
        store.create()
    synced = _SyncGetDict(threading.Barrier(2))
    synced.update(store._expires)
    store._expires = synced
    threads = [threading.Thread(target=store.create) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) <= 3                                # 检查-淘汰不原子时，两个线程淘汰同一个 token，上限失效
