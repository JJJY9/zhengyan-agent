"""出域/并发/限速回归：无动词持仓出域拦截、模拟盘跨进程净值认领、登录限速按账号分桶与反代文档。"""
import threading
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from advisor.api.app import _LoginThrottle, create_app
from advisor.contracts import ResearchRequest
from advisor.harness import HarnessDenied
from advisor.harness.egress import EgressGuard
from advisor.settings import ROOT
from advisor.strategy import StrategyBook
from advisor.strategy.store import StrategyStore
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service
from tests.test_phase4 import START, spec


# ---------- #2：无动词持仓出域 ----------

def _guard(policy):
    return EgressGuard(policy.egress.get("sensitive_patterns", []), policy.egress.get("trade_intent_patterns", []))


@pytest.mark.parametrize("text", ["我的600519成本价120元", "我的600519 1000股", "我600519 1000股成本120元"])
def test_verbless_holdings_detected(policy, text):
    with pytest.raises(HarnessDenied):
        _guard(policy).check_outbound(text)                # 「第一人称+代码+金额/股数」不带动词也要拦下


def test_public_mentions_not_flagged(policy):
    assert _guard(policy).check_outbound("复核 600999 半年报业绩与估值") is None   # 无第一人称不误伤


# ---------- #9：跨进程净值认领 ----------

class _FirstReadStore(StrategyStore):
    """首个 navs_of 让两线程同时读到同一快照；后写线程延迟插入，保证先到者先提交。"""

    def __init__(self, db, barrier):
        super().__init__(db)
        self._barrier = barrier
        self._tls = threading.local()

    def navs_of(self, card_id, version):
        out = super().navs_of(card_id, version)
        if threading.current_thread().name in ("a", "b") and not getattr(self._tls, "hit", False):
            self._tls.hit = True
            try:
                self._barrier.wait(timeout=1.0)
            except threading.BrokenBarrierError:
                pass
        return out

    def add_nav(self, p):
        if threading.current_thread().name == "b":         # 让「推进更晚日期」的进程后写
            time.sleep(0.05)
        return super().add_nav(p)


def test_cross_process_claims_nav_atomically(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    barrier = threading.Barrier(2)
    book_a = StrategyBook(svc.deps, _FirstReadStore(repo.db, barrier))
    book_b = StrategyBook(svc.deps, _FirstReadStore(repo.db, barrier))
    v = book_a.create(spec(max_positions=1), "owner", "跨进程并发推进的测试策略记录")
    book_a.start_paper(v.card_id, v.version, "owner", start=START)
    card = book_a.store.get(v.card_id, v.version)
    errors: list[Exception] = []

    def run(book, day):
        try:
            book.paper.step(card, day)
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(name="a", target=run, args=(book_a, START + timedelta(days=2)))
    t2 = threading.Thread(name="b", target=run, args=(book_b, START + timedelta(days=3)))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors
    navs = book_a.store.navs_of(v.card_id, v.version)
    assert [n.date for n in navs] == [START + timedelta(days=2), START + timedelta(days=3)]
    assert [n.seq for n in navs] == [1, 2]                 # 序号即认领：后写者必须重算
    opens = [t for t in book_a.store.list_trades(v.card_id, v.version) if t.side == "open"]
    assert len(opens) == 1                                  # 后一天基于前一天持仓，不能重复开仓


# ---------- #16：登录限速按账号分桶 ----------

def test_login_throttle_buckets_per_account():
    t = _LoginThrottle(limit=3, window_s=60)
    for _ in range(3):
        t.fail("1.2.3.4", "admin")
    assert t.blocked("1.2.3.4", "admin")
    assert not t.blocked("1.2.3.4", "guest")               # 不同账号不共享计数
    t.success("1.2.3.4", "admin")
    assert not t.blocked("1.2.3.4", "admin")


def test_spoofed_usernames_do_not_lock_correct_account(repo, policy):
    svc = make_service(repo, policy)
    with TestClient(create_app(deps=svc.deps, checkpointer=None, auth=("admin", "pw"))) as c:
        for _ in range(10):
            assert c.post("/api/login", json={"user": "attacker", "password": "x"}).status_code == 401
        r = c.post("/api/login", json={"user": "admin", "password": "pw"})
        assert r.status_code == 200                        # 陌生账号的爆破不能锁死正确账号（同一来源 IP）


def test_proxy_real_ip_documented():
    assert "FORWARDED_ALLOW_IPS" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "FORWARDED_ALLOW_IPS" in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
