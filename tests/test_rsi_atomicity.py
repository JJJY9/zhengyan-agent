"""RSI 发布/回滚原子性回归：并发双发布单一生效、发布与并发拒绝互斥、CAS 失败补偿、
回滚部分失败后可重试自愈、归档扫描不受 50 条列表上限截断。"""
import threading
from datetime import datetime, timedelta, timezone

import pytest

from advisor.rsi import RSIRunner, TransitionError
from advisor.rsi.models import Run
from tests.test_phase2 import make_service
from tests.test_rsi import RSILLM, seed_dossiers


@pytest.fixture
def rsi(repo, policy):
    seed_dossiers(repo)
    svc = make_service(repo, policy, RSILLM())
    return svc, RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)


def _approved(runner):
    runner.collect()
    issue = next(i for i in runner.ledger.list_issues("S1") if i.source["ticker"] == "600001")
    run = runner.propose("S1", issue.id)
    run = runner.evaluate(run.run_id)
    assert run.status == "review", run.reject_reason
    return runner.approve(run.run_id, "owner"), issue


def _in_threads(fns):
    out = []
    def wrap(fn):
        try:
            out.append(("ok", fn()))
        except Exception as e:      # noqa: BLE001
            out.append(("err", e))
    ts = [threading.Thread(target=wrap, args=(f,)) for f in fns]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def test_concurrent_double_release_single_effect(rsi, monkeypatch):
    svc, runner = rsi
    run, issue = _approved(runner)
    led = runner.ledger
    barrier = threading.Barrier(2)
    real_save = led.save_run

    def sync_save(r, expect_status=None):
        if expect_status == "approved":          # 两个请求都通过 _get、同时尝试占位
            try:
                barrier.wait(timeout=3)
            except threading.BrokenBarrierError:
                pass
        return real_save(r, expect_status)

    monkeypatch.setattr(led, "save_run", sync_save)
    results = _in_threads([lambda: runner.release(run.run_id, "owner"),
                           lambda: runner.release(run.run_id, "owner")])
    monkeypatch.setattr(led, "save_run", real_save)
    assert sum(1 for s, _ in results if s == "ok") == 1
    assert led.active_version("lessons") == 2
    assert len(led.list_assets("lessons")) == 2                       # 只有一个新版本，失败方不留孤儿资产
    assert sum(1 for e in led.list_events(limit=200) if e["action"] == "release") == 1
    assert led.get_run(run.run_id).status == "released"
    assert led.get_issues([issue.id])[0].status == "addressed"


def test_release_vs_reject_leaves_no_partial_state(rsi, monkeypatch):
    svc, runner = rsi
    run, issue = _approved(runner)
    led = runner.ledger
    real_append = led.append_asset
    reject_err = {}

    def hooked(*a, **k):
        # 放慢发布方：先让并发拒在旧实现的时间窗口里完整落库
        def do_reject():
            try:
                runner.reject(run.run_id, "reviewer", "并发拒绝：不发布这一版")
            except Exception as e:      # noqa: BLE001
                reject_err["e"] = e
        t = threading.Thread(target=do_reject)
        t.start()
        t.join()
        return real_append(*a, **k)

    monkeypatch.setattr(led, "append_asset", hooked)
    got = _in_threads([lambda: runner.release(run.run_id, "owner")])
    monkeypatch.setattr(led, "append_asset", real_append)
    final = led.get_run(run.run_id)
    if final.status == "released":            # 发布会话赢：拒绝方必须失败且没有任何副作用
        assert isinstance(reject_err.get("e"), TransitionError)
        assert got[0][0] == "ok"
        assert led.active_version("lessons") == 2 and len(led.list_assets("lessons")) == 2
        assert led.get_issues([issue.id])[0].status == "addressed"
    else:                                     # 拒绝赢：发布方必须失败且不留下资产/误标
        assert final.status == "rejected"
        assert got[0][0] == "err"
        assert led.active_version("lessons") == 1 and len(led.list_assets("lessons")) == 1
        assert led.get_issues([issue.id])[0].status == "open"


def test_release_cas_failure_compensates(rsi, monkeypatch):
    svc, runner = rsi
    run, issue = _approved(runner)
    led = runner.ledger
    monkeypatch.setattr(led, "activate", lambda kind, expected, new: False)   # 模拟指针已被并发修改
    with pytest.raises(TransitionError, match="并发修改"):
        runner.release(run.run_id, "owner")
    assert len(led.list_assets("lessons")) == 1                # 未激活资产已删除
    assert led.active_version("lessons") == 1
    assert led.get_run(run.run_id).status == "approved"        # 状态占位退回，可重试
    assert led.get_issues([issue.id])[0].status == "open"      # 坏案例未被误标


def test_rollback_cas_conflict_keeps_state(rsi, monkeypatch):
    svc, runner = rsi
    run, issue = _approved(runner)
    led = runner.ledger
    runner.release(run.run_id, "owner")
    monkeypatch.setattr(led, "activate", lambda kind, expected, new: False)
    with pytest.raises(TransitionError, match="并发修改"):
        runner.rollback("lessons", "owner", "回滚遇到并发修改的测试")
    assert led.active_version("lessons") == 2                  # 指针未动
    assert led.get_run(run.run_id).status == "released"
    assert led.get_issues([issue.id])[0].status == "addressed"


def test_rollback_retry_heals_partial_state(rsi, monkeypatch):
    svc, runner = rsi
    run, issue = _approved(runner)
    led = runner.ledger
    runner.release(run.run_id, "owner")
    real_move = runner._move

    def flaky_move(r, status, *a, **k):
        if status == "rolled_back":            # 模拟回滚在标记运行记录前中断
            raise TransitionError("模拟中断")
        return real_move(r, status, *a, **k)

    monkeypatch.setattr(runner, "_move", flaky_move)
    runner.rollback("lessons", "owner", "第一次回滚部分失败的测试")
    assert led.active_version("lessons") == 1                  # 指针已回退
    assert led.get_run(run.run_id).status == "released"        # 账本尚未对齐
    monkeypatch.setattr(runner, "_move", real_move)
    with pytest.raises(TransitionError, match="基线"):
        runner.rollback("lessons", "owner", "重试回滚以对齐账本记录")   # 重试被基线拒绝，但会先自愈
    assert led.get_run(run.run_id).status == "rolled_back"
    assert led.get_issues([issue.id])[0].status == "open"


def test_activate_is_compare_and_set(repo, policy):
    from advisor.rsi.ledger import Ledger
    led = Ledger(repo.db)
    led.bootstrap(policy)
    barrier = threading.Barrier(2)
    real = led.activate

    def synced(kind, expected, new):
        barrier.wait(timeout=3)
        return real(kind, expected, new)

    out = _in_threads([lambda: synced("lessons", 1, 2), lambda: synced("lessons", 1, 3)])
    wins = [r for s, r in out if s == "ok" and r is True]
    assert len(wins) == 1                                  # 同一 expected 只有一个 CAS 成功
    assert led.active_version("lessons") in (2, 3)


def test_release_archives_old_runs_beyond_list_limit(rsi):
    svc, runner = rsi
    run, _ = _approved(runner)
    led = runner.ledger
    now = datetime.now(timezone.utc)
    old_ids = []
    for i in range(60):                        # 比当前运行更早的 60 条已发布运行
        rid = f"rsi_old_{i:02d}"
        old_ids.append(rid)
        ledger_run = Run(run_id=rid, scenario="S1", kind="lessons", issue_ids=[], base_version=1, content={},
                         change_reason="历史运行占位记录（归档扫描测试）", direction=f"S1:old:{i}",
                         status="released", released_version=1, created_at=now - timedelta(days=1) + timedelta(seconds=i),
                         updated_at=now - timedelta(days=1) + timedelta(seconds=i))
        led.insert_run(ledger_run)
    runner.release(run.run_id, "owner")
    assert all(led.get_run(rid).status == "archived" for rid in old_ids)   # 超过 50 条也不能漏归档
