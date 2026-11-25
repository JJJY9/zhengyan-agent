"""订阅扫描与复核回归：复权基准、价格指纹去重/收敛、公告判定、周期归一、扫描失败、同步与忽略校验。"""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import Dossier, QualityResult, ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.runner import ResearchService
from advisor.personal.officer import assess
from advisor.scheduler import Scheduler
from advisor.services import Services
from advisor.watch import WatchError
from advisor.watch.models import Alert
from advisor.watch.scanner import Watcher
from advisor.watch.store import WatchStore
from tests.fakes import CODE, FakeLLM, FakeSource, FakeWebTools, _row

START = date(2026, 10, 18)


def _services(repo, policy, source):
    deps = build_deps(repo, FakeLLM(), policy, source=source, web_tools=FakeWebTools())
    return Services(deps, ResearchService(deps, InMemorySaver()))


def _research(svcs, as_of):
    return svcs.research.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of, route="standard"),
                               background=False)


def _alert(store, fingerprint="fp"):
    a = Alert(id="al_" + fingerprint, ticker=CODE, name="示例酒业", dossier_key=CODE, dossier_version=1,
              fingerprint=fingerprint, events=[], affected=[], created_at=datetime.now(timezone.utc))
    store.add_alert(a)
    return a


class _FlatSource(FakeSource):
    split = False

    def daily(self, code, start, end):
        v = "500" if self.split else "1000"
        return [dict(r, close=v) for r in super().daily(code, start, end)]


def test_price_move_uses_common_adjustment_basis(repo, policy):
    src = _FlatSource()
    svcs = _services(repo, policy, src)
    _research(svcs, START)
    svcs.watch.store.subscribe(CODE, "示例酒业", price_move_pct=10)
    src.split = True                      # 十送十后前复权序列整体重算：1000 → 500
    assert svcs.watch.scan(CODE, date(2026, 10, 25)) is None   # 同一复权基准 500 → 500，不是 -50%


class _StepSource(FakeSource):
    def daily(self, code, start, end):
        cut = date(2026, 10, 19)
        return [dict(r, close="1200" if date.fromisoformat(r["date"]) >= cut else "1000")
                for r in super().daily(code, start, end)]


def test_price_alert_dedupes_same_version_and_supersedes(repo, policy):
    src = _StepSource()
    svcs = _services(repo, policy, src)
    _research(svcs, START)
    store = svcs.watch.store
    store.subscribe(CODE, "示例酒业", price_move_pct=10)
    a1 = svcs.watch.scan(CODE, date(2026, 10, 25))
    a2 = svcs.watch.scan(CODE, date(2026, 10, 26))
    assert a1 is not None and a2 is not None and a1.id == a2.id       # 同一批变化只提醒一次
    assert len(store.list_alerts(CODE, "open")) == 1
    _research(svcs, date(2026, 10, 27))                                 # 复核产生 v2，基准更新
    assert svcs.watch.scan(CODE, date(2026, 10, 27)) is None
    assert store.get_alert(a1.id).status == "dismissed"                 # 旧版本提醒自动收敛


class _AnnSource(FakeSource):
    extra = False

    def announcements(self, code, start, end):
        rows = super().announcements(code, start, end)
        if self.extra:                 # 与 as_of 同日、但在研究之后披露
            rows = rows + [{"name": "示例酒业", "title": "关于签订重大合同的公告", "date": "2026-10-18",
                            "url": "http://www.cninfo.com.cn/x/9"}]
        return rows


def test_same_day_announcement_after_research_alerted(repo, policy):
    src = _AnnSource()
    svcs = _services(repo, policy, src)
    _research(svcs, START)
    svcs.watch.store.subscribe(CODE, "示例酒业")
    src.extra = True
    alert = svcs.watch.scan(CODE, date(2026, 10, 19))
    assert alert is not None
    ann = next(e for e in alert.events if e.kind == "announcement")
    assert "签订重大合同" in ann.detail


class _OldAnnSource(FakeSource):
    as_of = date.today() - timedelta(days=450)

    def income_rows(self, code):           # 补齐更早年度，保证旧时点也能算出同比
        b = 10 ** 9
        return super().income_rows(code) + [
            _row("2024-12-31", "2025-04-17", 160 * b, 80 * b),
            _row("2024-09-30", "2024-10-28", 120 * b, 60 * b),
            _row("2024-06-30", "2024-08-14", 85 * b, 42 * b),
            _row("2024-03-31", "2024-04-29", 50 * b, 25 * b),
        ]

    def announcements(self, code, start, end):
        rows = [
            {"name": "示例酒业", "title": "旧重组公告",
             "date": (self.as_of + timedelta(days=10)).isoformat(), "url": "http://www.cninfo.com.cn/x/old"},
            {"name": "示例酒业", "title": "近期公告", "date": (date.today() - timedelta(days=5)).isoformat(),
             "url": "http://www.cninfo.com.cn/x/new"},
        ]
        return [r for r in rows if start <= date.fromisoformat(r["date"]) <= end]


def test_announcements_cover_more_than_400_days(repo, policy):
    src = _OldAnnSource()
    svcs = _services(repo, policy, src)
    _research(svcs, src.as_of)
    svcs.watch.store.subscribe(CODE, "示例酒业")
    alert = svcs.watch.scan(CODE, date.today())
    assert alert is not None
    ann = next(e for e in alert.events if e.kind == "announcement")
    assert len(ann.evidence_ids) == 2                    # 400 天以外的旧公告也要覆盖


def _dossier(evidence_ids):
    return Dossier(key=CODE, version=1, ticker=CODE, name="示例酒业", as_of=START, question="q", route="standard",
                   status="committed", claims=[], evidence_ids=evidence_ids, quality=QualityResult(decision="pass"),
                   task_id="t", created_at=datetime.now(timezone.utc))


class _H:
    def __init__(self, card):
        self.card = card

    def call(self, name, **kw):
        return SimpleNamespace(cards=[self.card])


def test_new_report_supports_compact_periods(repo):
    from advisor.contracts import Metric
    from advisor.evidence import EvidenceFactory
    f = EvidenceFactory(lambda h, c: None)
    old = f.make("old", level="A", source_type="filing", title="财报", origin_id="filing:x:20260630",
                 as_of=START, excerpt="", metrics=[Metric(name="营业收入", field="OPERATE_INCOME", value=Decimal(1),
                                                         unit="元", period="20260630", basis="cumulative")],
                 skill="financials.statements", skill_version="1")
    new = f.make("new", level="A", source_type="filing", title="财报", origin_id="filing:x:2026-09-30",
                 as_of=START, published_at=datetime(2026, 10, 28, tzinfo=timezone.utc), excerpt="",
                 metrics=[Metric(name="营业收入", field="OPERATE_INCOME", value=Decimal(2), unit="元",
                                 period="2026-09-30", basis="cumulative")],
                 skill="financials.statements", skill_version="1")
    watcher = Watcher.__new__(Watcher)
    found = watcher._new_report(_H(new), _dossier([old.id]), {old.id: old}, date(2026, 10, 30), None)
    assert found is not None                              # 紧凑期号与连字符期号可以正确比较


def test_freshness_marks_scan_failure_unknown():
    fit = assess(_dossier([]), [], None, subscribed=True, open_alerts=[],
                 scan_note="无变化（new_report 取数失败：网络错误）")
    fresh = next(c for c in fit.checks if c.key == "freshness")
    assert fresh.status == "unknown" and "没能取到数据" in fresh.detail


def test_dismiss_rejects_non_open_alert(repo):
    store = WatchStore(repo.db)
    a = _alert(store, "fp_dismiss")
    assert store.try_start_review(a.id)                    # status=reviewing
    watcher = Watcher(SimpleNamespace(repo=repo), store)
    with pytest.raises(WatchError, match="待处理"):
        watcher.dismiss(a.id, "不想看了")


def test_scheduler_syncs_and_recovers_missing_task(repo, policy):
    svcs = _services(repo, policy, FakeSource())
    store = svcs.watch.store
    done = _alert(store, "fp_done")
    store.try_start_review(done.id)
    repo.create_task("t_done", {"question": "q"})
    repo.update_task("t_done", status="done", result={"dossier_key": CODE, "version": 2})
    store.update_alert(done.id, review_task_id="t_done")
    orphan = _alert(store, "fp_orphan")
    store.try_start_review(orphan.id)
    store.update_alert(orphan.id, review_task_id="t_missing")

    Scheduler(svcs, 0).tick()                              # 定时任务本身就要回写，而不是等网页访问

    assert store.get_alert(done.id).status == "resolved" and store.get_alert(done.id).resolved_version == 2
    o = store.get_alert(orphan.id)
    assert o.status == "open" and "缺失" in o.note          # 任务丢失不能永久卡在复核中
