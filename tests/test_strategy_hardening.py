"""策略卡状态机回归：回滚顺序、批准复核门控、晋级补偿、编号竞态、并发推进、成交补偿、step 去重键。"""
import threading
from datetime import timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from advisor.api.app import create_app
from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.services import Services
from advisor.strategy import StrategyBook, StrategyError
from advisor.strategy.store import StrategyStore, spec_hash
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, as_of
from tests.test_phase2 import make_service
from tests.test_phase4 import START, run_days, spec


@pytest.fixture
def services(service):
    service.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    return Services(service.deps, service)


def _prepare(book: StrategyBook, version_spec: dict, n: int, card_id: str | None = None):
    v = book.create(spec(**version_spec), "owner", f"测试策略版本第 {n} 次修改", card_id=card_id)
    book.start_paper(v.card_id, v.version, "owner", start=START)
    run_days(book, START, 10)
    book.submit(v.card_id, v.version, "owner")
    return v


# ---------- 回滚按晋级顺序 ----------

def test_rollback_restores_previous_promoted_by_order(services):
    book = services.strategy
    v1 = _prepare(book, {}, 1)
    assert book.approve(v1.card_id, v1.version, "owner").status == "promoted"
    v2 = _prepare(book, {"stop_loss_pct": 29}, 2, card_id=v1.card_id)
    v3 = _prepare(book, {"stop_loss_pct": 28}, 3, card_id=v1.card_id)
    assert book.approve(v3.card_id, v3.version, "owner").status == "promoted"
    assert book.approve(v2.card_id, v2.version, "owner").status == "promoted"     # 乱序：先批 v3 再批 v2
    book.rollback(v1.card_id, "owner", "测试回滚到上一个在用版本")
    assert book.promoted(v1.card_id).version == 3     # v2 晋级前的在用版本是 v3，不是版本号更小的 v1


# ---------- 批准时复跑门控 ----------

class _CrashSource(FakeSource):
    crash = False

    def daily(self, code, start, end):
        rows = super().daily(code, start, end)
        if not self.crash:
            return rows
        return [dict(r, close=str(Decimal(r["close"]) / 2)) if r["date"] >= "2026-10-29" else r for r in rows]


def test_approve_reruns_gates(repo, policy):
    src = _CrashSource()
    deps = build_deps(repo, FakeLLM(), policy, source=src, web_tools=FakeWebTools())
    svcs = Services(deps, ResearchService(deps, InMemorySaver()))
    svcs.research.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                        background=False)
    book = svcs.strategy
    v = book.create(spec(max_drawdown_limit=5), "owner", "批准前的门控复核测试记录")
    book.start_paper(v.card_id, v.version, "owner", start=START)
    run_days(book, START, 10)
    assert book.submit(v.card_id, v.version, "owner").status == "review"
    src.crash = True
    run_days(book, START + timedelta(days=10), 2)          # review 期间继续推进，回撤恶化
    with pytest.raises(StrategyError, match="已失效"):
        book.approve(v.card_id, v.version, "owner")
    assert book.store.get(v.card_id, v.version).status == "review"    # 留在待晋级，可继续模拟


# ---------- 晋级失败补偿 ----------

def test_approve_restores_incumbent_on_failure(services, monkeypatch):
    book = services.strategy
    v1 = _prepare(book, {}, 1)
    book.approve(v1.card_id, v1.version, "owner")
    v2 = _prepare(book, {"stop_loss_pct": 29}, 2, card_id=v1.card_id)
    real_save = book.store.save

    def flaky(v, expect_status=None):
        if v.version == 2 and v.status == "promoted":
            raise RuntimeError("db 在两次写入间故障")
        return real_save(v, expect_status)

    monkeypatch.setattr(book.store, "save", flaky)
    with pytest.raises(RuntimeError):
        book.approve(v2.card_id, v2.version, "owner")
    assert book.promoted(v1.card_id).version == 1          # 候选晋级失败：在用版本恢复
    assert book.store.get(v1.card_id, 2).status == "review"


def test_double_approve_does_not_retire_winner(services, monkeypatch):
    book = services.strategy
    v1 = _prepare(book, {}, 1)
    book.approve(v1.card_id, v1.version, "owner")
    v2 = _prepare(book, {"stop_loss_pct": 29}, 2, card_id=v1.card_id)
    stale = book.store.get(v2.card_id, 2)                   # 另一个请求读到的旧快照（review）
    assert book.approve(v2.card_id, 2, "owner").status == "promoted"
    real_get = book.store.get
    monkeypatch.setattr(book.store, "get",
                        lambda cid, ver=None: stale if (cid, ver) == (v2.card_id, 2) else real_get(cid, ver))
    # 必须命中同版本守卫：若先去退役再用旧快照晋级，乐观锁也会抛「状态被改变」，
    # 但那是把已晋级版本退役后才恢复的坏路径，报错文案不同
    with pytest.raises(StrategyError, match="已经是晋级版本"):
        book.approve(v2.card_id, 2, "owner")
    assert book.promoted(v2.card_id).version == 2           # 已晋级的版本不能被后到者退役


# ---------- 卡号/版本号并发分配 ----------

def test_create_retries_on_card_id_collision(services, monkeypatch):
    book = services.strategy
    seq = iter(["SC-001", "SC-001", "SC-002"])               # 两个请求都先算出 SC-001
    monkeypatch.setattr(book.store, "next_card_id", lambda: next(seq))
    v1 = book.create(spec(), "owner", "并发新建策略卡第一号测试")
    v2 = book.create(spec(stop_loss_pct=29), "owner", "并发新建策略卡第二号测试")
    assert (v1.card_id, v2.card_id) == ("SC-001", "SC-002")


def test_next_card_id_uses_max_numeric(repo):
    store = StrategyStore(repo.db)
    store.cards.insert_many([{"card_id": "SC-002"}, {"card_id": "SC-010"}, {"card_id": "OTHER"}])
    # 按最大编号 +1 而不是按张数 +1：删过卡或编号不连续时，按张数会撞上已用编号
    assert store.next_card_id() == "SC-011"


def test_create_retries_on_version_collision(services, monkeypatch):
    book = services.strategy
    v1 = book.create(spec(), "owner", "版本竞态测试的基础版本记录")
    other = spec(stop_loss_pct=29)
    from advisor.strategy.models import StrategyVersion
    from datetime import datetime, timezone
    book.store.insert(StrategyVersion(**other.model_dump(), card_id=v1.card_id, version=2, parent_version=1,
                                      change_reason="另一标签页抢先修订的版本", sha256=spec_hash(other),
                                      created_by="other", created_at=datetime.now(timezone.utc)))
    real_get, calls = book.store.get, {"n": 0}

    def stale_get(card_id, version=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return v1                                       # 第一次读到旧父版本 v1
        return real_get(card_id, version)

    monkeypatch.setattr(book.store, "get", stale_get)
    v3 = book.create(spec(stop_loss_pct=28), "owner", "并发修订同一策略卡的测试", card_id=v1.card_id)
    assert (v3.version, v3.parent_version) == (3, 2)        # 撞版本号后重读最新版本再分配


# ---------- 并发推进不同日期 ----------

class _BarrierStore(StrategyStore):
    def __init__(self, db):
        super().__init__(db)
        self.barrier = threading.Barrier(2)

    def navs_of(self, card_id, version):
        result = super().navs_of(card_id, version)
        if threading.current_thread().name in ("t1", "t2"):
            try:
                self.barrier.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
        return result


def test_concurrent_steps_on_different_days_are_serialized(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    store = _BarrierStore(repo.db)
    book = StrategyBook(svc.deps, store)
    v = book.create(spec(max_positions=1), "owner", "并发推进不同日期的测试记录")
    book.start_paper(v.card_id, v.version, "owner", start=START)
    v = book.store.get(v.card_id, v.version)
    errors = []

    def run(name, day):
        try:
            book.paper.step(v, day)
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(name="t1", target=run, args=("t1", START + timedelta(days=2)))
    t2 = threading.Thread(name="t2", target=run, args=("t2", START + timedelta(days=3)))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    navs = store.navs_of(v.card_id, v.version)
    assert [n.date for n in navs] == [START + timedelta(days=2), START + timedelta(days=3)]
    opens = [t for t in store.list_trades(v.card_id, v.version) if t.side == "open"]
    assert len(opens) == 1                                # 后一天基于前一天持仓，不能重复开仓


# ---------- 成交写入失败可恢复 ----------

def test_step_retries_trades_after_write_failure(services, monkeypatch):
    book = services.strategy
    v = book.create(spec(), "owner", "成交写入失败的补偿测试记录")
    book.start_paper(v.card_id, v.version, "owner", start=START)
    v = book.store.get(v.card_id, v.version)
    real = book.store.add_trades
    state = {"n": 0}

    def flaky(trades):
        if state["n"] == 0:
            state["n"] += 1
            raise ConnectionError("成交集合瞬时不可用")
        return real(trades)

    monkeypatch.setattr(book.store, "add_trades", flaky)
    with pytest.raises(ConnectionError):
        book.paper.step(v, START)
    assert book.store.navs_of(v.card_id, v.version) == []     # 净值点撤销，允许重试
    book.paper.step(v, START)                                  # 重试补齐净值与成交
    assert book.store.navs_of(v.card_id, v.version) and book.store.list_trades(v.card_id, v.version)


# ---------- step 作业去重键包含日期 ----------

def test_step_job_key_includes_day(service, monkeypatch):
    import time
    with TestClient(create_app(deps=service.deps)) as c:
        seen = []

        def fake_step(day=None, log=lambda m: None):
            seen.append(day)
            time.sleep(0.3)
            return []

        monkeypatch.setattr(c.app.state.services.strategy, "step_all", fake_step)
        r1 = c.post("/api/strategy/step?day=2026-11-20")
        r2 = c.post("/api/strategy/step?day=2026-11-21")
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["job_id"] != r2.json()["job_id"]     # 不同日期各自排队，不能被静默丢弃
