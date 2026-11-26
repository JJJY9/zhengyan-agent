"""窗口回归修复：出域误伤/漏检、净值 seq 空洞、登录限速 IP 总闸与清理、扫描备注、
发布簿记冲突补偿、坏案例确认过滤、注入门控误伤。"""
from datetime import date, datetime, timezone

import pytest

from advisor.api.app import _LoginThrottle
from advisor.contracts import ResearchRequest
from advisor.harness import HarnessDenied, SkillFailed
from advisor.harness.egress import EgressGuard
from advisor.rsi import RSIRunner, TransitionError
from advisor.rsi.gates import overfit_gate
from advisor.rsi.issues import _issue
from advisor.rsi.ledger import Ledger
from advisor.rsi.models import Run
from advisor.services import Services
from advisor.strategy import StrategyBook
from advisor.strategy.store import StrategyStore
from advisor.watch.models import WatchEvent
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service
from tests.test_phase4 import START, spec
from tests.test_rsi import RSILLM, seed_dossiers


def _guard(policy):
    return EgressGuard(policy.egress.get("sensitive_patterns", []), policy.egress.get("trade_intent_patterns", []))


# ---------- #1：公开提问不能被当成个人信息 ----------

@pytest.mark.parametrize("text", [
    "我想问600519目标价100元合理吗",
    "我们预计600519 2025年分红每股3元，合理吗",
    "我看了600519 2024年三季报营收100万元",
    "帮我直接买入 600999",
    "帮我买100股 600999，现在合适吗",       # 交易意图提问：由边界提示处理，不是个人信息
])
def test_public_questions_not_flagged(policy, text):
    assert _guard(policy).check_outbound(text) is None


# ---------- #2：金额/动词在代码前的持仓写法必须拦下 ----------

@pytest.mark.parametrize("text", [
    "我120元买入的600519现在怎么办",
    "我120元买的600519",
    "我买600519 1000股",
    "我的600519成本价120元",                 # 原有三种写法不得回退
    "我的600519 1000股",
    "我600519 1000股成本120元",
])
def test_personal_cost_phrasings_detected(policy, text):
    with pytest.raises(HarnessDenied):
        _guard(policy).check_outbound(text)


# ---------- #3/#10：seq 空洞与补偿删除 ----------

def _paper_book(repo, policy, store_cls=StrategyStore):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    book = StrategyBook(svc.deps, store_cls(repo.db))
    v = book.create(spec(max_positions=1), "owner", "窗口回归的模拟盘测试策略记录")
    book.start_paper(v.card_id, v.version, "owner", start=START)
    return book, book.store.get(v.card_id, v.version)


def test_seq_hole_does_not_stall_paper(repo, policy):
    book, card = _paper_book(repo, policy)
    for day in (START, START.replace(day=START.day)):   # 用不同日期推进三天
        pass
    from datetime import timedelta
    for d in (START, START + timedelta(days=1), START + timedelta(days=2)):
        book.paper.step(card, d)
    book.store.delete_nav(card.card_id, card.version, START + timedelta(days=1))   # 人为制造 seq 空洞
    book.paper.step(card, START + timedelta(days=3))    # 取 max(seq)+1：空洞不应导致永久撞索引
    seqs = [n.seq for n in book.store.navs_of(card.card_id, card.version)]
    assert seqs == [1, 3, 4]


class _HookStore(StrategyStore):
    """成交写入时先执行一次外部动作再按需失败（模拟失败窗口里的并发推进）。"""

    def __init__(self, db):
        super().__init__(db)
        self.hook = None

    def add_trades(self, trades):
        if self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
            raise ConnectionError("成交集合瞬时不可用")
        return super().add_trades(trades)


def test_trades_failure_with_later_nav_is_reported_not_deleted(repo, policy):
    from datetime import timedelta
    store = _HookStore(repo.db)
    book, card = _paper_book(repo, policy, store_cls=lambda db: store)
    book.paper.step(card, START)
    # 失败窗口内另一进程基于今天的净值点推进了更晚日期
    book.store.hook = lambda: book.paper._step_once(card, START + timedelta(days=2))
    with pytest.raises(Exception) as ei:
        book.paper.step(card, START + timedelta(days=1))
    assert "人工核对" in str(ei.value)                    # 有更晚净值点：不能删除留空洞，显式报出
    navs = book.store.navs_of(card.card_id, card.version)
    assert [n.date for n in navs] == [START, START + timedelta(days=1), START + timedelta(days=2)]
    book.paper.step(card, START + timedelta(days=3))      # 后续推进不受影响


# ---------- #4：登录限速 IP 总闸与过期清理 ----------

def test_login_throttle_ip_aggregate_cap():
    t = _LoginThrottle(limit=2, window_s=60)
    for i in range(10):                                   # 换着用户名喷洒：IP 总闸 = 2×5
        t.fail("9.9.9.9", f"user{i}")
    assert t.blocked("9.9.9.9", "anyone")
    assert not t.blocked("8.8.8.8", "anyone")


def test_login_throttle_prunes_expired_buckets():
    now = [0.0]
    t = _LoginThrottle(limit=5, window_s=60, now=lambda: now[0])
    t.PRUNE_AT = 50
    for i in range(120):
        t.fail("1.1.1.1", f"u{i}")
    assert len(t._fails) <= 55                            # 超过阈值触发全量清理，键不会无界增长
    now[0] = 10_000.0
    t._prune_all()
    assert len(t._fails) == 0


# ---------- #5：有提醒时取数失败也进扫描记录 ----------

def test_scan_notes_survive_when_alert_created(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    services = Services(svc.deps, svc)
    services.watch.store.subscribe(CODE, "示例酒业", 10)

    def boom(*a, **k):
        raise SkillFailed("数据源不可用")

    services.watch._new_report = boom
    services.watch._price_move = boom
    services.watch._announcements = lambda *a, **k: (WatchEvent(kind="announcement", title="测试公告"), [], [])
    alert = services.watch.scan(CODE, today=date(2026, 10, 30))
    assert alert is not None
    note = services.watch.store.get_sub(CODE).last_scan_note
    assert "取数失败" in note                             # 有提醒产出时失败原因不能被覆盖


# ---------- #6/#7/#8：发布簿记 ----------

def _approved_s1(repo, policy):
    seed_dossiers(repo)
    svc = make_service(repo, policy, RSILLM())
    runner = RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)
    runner.collect()
    issue = next(i for i in runner.ledger.list_issues("S1") if i.source["ticker"] == "600001")
    run = runner.propose("S1", issue.id)
    run = runner.evaluate(run.run_id)
    assert run.status == "review", run.reject_reason
    return runner, runner.approve(run.run_id, "owner"), issue


def test_release_conflict_rolls_back_pointer(repo, policy, monkeypatch):
    runner, run, issue = _approved_s1(repo, policy)
    led = runner.ledger
    real = led.save_run

    def stale(r, expect_status=None):
        if expect_status == "released":                   # 模拟并发回滚/归档改变运行状态
            return False
        return real(r, expect_status)

    monkeypatch.setattr(led, "save_run", stale)
    with pytest.raises(TransitionError, match="冲突"):
        runner.release(run.run_id, "owner")
    assert led.active_version("lessons") == 1             # 指针已回退
    assert len(led.list_assets("lessons")) == 1           # 未激活资产已删除
    assert not any(e["action"] == "release" for e in led.list_events(limit=100))   # 不能谎报发布成功
    assert led.get_issues([issue.id])[0].status == "open"  # 坏案例也不能被标记


def test_resolved_issues_only_from_scored_evals(repo, policy):
    svc = make_service(repo, policy)
    runner = RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)
    runner.ledger.bootstrap(svc.deps.policy)
    now = datetime.now(timezone.utc)
    q = "复核业绩"
    good_row = {"question": q, "label": "deep", "route": "deep", "source": "model",
                "probs": {"quick": 0.05, "standard": 0.05, "deep": 0.9}, "abstain": False, "correct": False}
    bad_row = {"question": q, "label": "quick", "route": "deep", "source": "model",
               "probs": {"quick": 0.1, "standard": 0.45, "deep": 0.45}, "abstain": False, "correct": False}
    for i in range(5):                                    # 最新 5 条全是失败的评测：不算数
        repo.db.eval_runs.insert_one({"run_id": f"rf_{i}", "type": "routing", "status": "failed",
                                      "created_at": now, "rows": [good_row]})
    repo.db.eval_runs.insert_one({"run_id": "rd", "type": "routing", "status": "done",
                                  "created_at": now - __import__("datetime").timedelta(minutes=10), "rows": [bad_row]})
    issue = _issue("S2", "route_mislabel", "q 应走 quick", ["证据"], "判别器校准测试假设", {"question": q}, q)
    runner.ledger.add_issue(issue)
    base = runner.ledger.active_asset("routing")
    run = Run(run_id="rsi_resolved", scenario="S2", kind="routing", issue_ids=[issue.id], base_version=base.version,
              content=dict(base.content), change_reason="坏案例确认过滤的测试候选", direction="S2:test",
              created_at=now, updated_at=now)
    assert runner._resolved_issues(run) == []             # failed 评测里「判对」的行不能把坏案例标已修复


# ---------- #9：注入门控不误伤合法文案 ----------

def _card(lesson: str) -> dict:
    return {"id": "L-001", "title": "方法卡", "applies_to": "任何公司", "tags": ["核验"], "lesson": lesson,
            "must_check": [], "invalid_if": "", "source": "", "rev": 1}


def test_gate_allows_legit_rule_wording(repo):
    ledger = Ledger(repo.db)
    now = datetime.now(timezone.utc)
    run = Run(run_id="rsi_gate_ok", scenario="S1", kind="lessons", issue_ids=[], base_version=1, content={},
              diff={"cards": [_card("不要忽略审计规则与监管要求，逐项核对披露口径")]},
              change_reason="合法方法卡的注入门控测试", direction="S1:test", created_at=now, updated_at=now)
    g = overfit_gate(run, {}, ledger)
    assert g.passed                                       # 「规则/要求」不是注入词：严格模式只认忽略…指令家族
