"""CLI 语义回归：启动回收只属服务进程、复核同步执行、--yes 不越复核门、扫描失败可见、
对照评测自动批准、无前任不可回滚、未来时点拒绝。"""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import advisor.cli as cli_mod
from advisor.cli import app
from advisor.contracts import ResearchRequest
from advisor.harness import SkillFailed
from advisor.services import Services
from advisor.strategy import StrategyError
from advisor.watch.scanner import Watcher
from advisor.watch.store import WatchStore
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service
from tests.test_phase4 import START, run_days, spec


# ---------- #8：启动回收只属服务进程 ----------

def test_cli_services_do_not_recover_other_process_tasks(repo, policy):
    svc = make_service(repo, policy)
    repo.create_task("t_running", {"question": "复核 600999"})
    repo.db.tasks.update_one({"task_id": "t_running"},
                             {"$set": {"status": "running",
                                       "updated_at": datetime.now(timezone.utc) - timedelta(seconds=3600)}})
    Services(svc.deps, svc)                              # CLI 构造路径：不能误杀其它进程的在途任务
    assert repo.get_task("t_running")["status"] == "running"
    Services(svc.deps, svc, recover_stale=True)          # 服务进程启动路径：照常回收
    assert repo.get_task("t_running")["status"] == "failed"


# ---------- #6：复核任务同步执行 ----------

def _alert(repo, status="open"):
    store = WatchStore(repo.db)
    store.alerts.insert_one({"id": "al_sync", "ticker": CODE, "name": "示例酒业", "dossier_key": CODE,
                             "dossier_version": 1, "fingerprint": "f-sync", "events": [], "affected": [],
                             "status": status, "created_at": datetime.now(timezone.utc)})
    return store


def test_review_can_run_synchronously(repo, policy):
    store = _alert(repo)
    watcher = Watcher(make_service(repo, policy).deps, store)

    class _Svc:
        def __init__(self):
            self.was_background = None

        def start(self, req, background=True, review_focus=None):
            self.was_background = background
            return "t_x"

    svc = _Svc()
    assert watcher.start_review("al_sync", svc, background=False) == "t_x"
    assert svc.was_background is False                   # CLI 进程退出会带走线程池：必须前台同步跑


# ---------- #7：--yes 不越过复核门 ----------

def _fake_research(repo, interrupt):
    class _R:
        def __init__(self):
            self.deps = SimpleNamespace(repo=repo)
            self.resumed: list[bool] = []

        def start(self, req, background=False):
            repo.create_task("t_cli", req.model_dump(mode="json"))
            repo.update_task("t_cli", status="waiting_approval", interrupt=interrupt)
            return "t_cli"

        def resume(self, tid, approve, background=False, **kw):
            self.resumed.append(approve)
            repo.update_task(tid, status="failed", error="测试结束")

    return _R()


def test_yes_does_not_auto_approve_review_gate(repo, monkeypatch):
    fake = _fake_research(repo, {"type": "review", "message": "存在高后果且未解决的争议", "items": ["争议 A"]})
    monkeypatch.setattr(cli_mod, "_services", lambda: SimpleNamespace(research=fake))
    res = CliRunner().invoke(app, ["research", "测试问题内容", "--yes"], input="n\n")
    assert fake.resumed == [False]                       # 高后果复核门必须人工确认，--yes 只对 consent 生效
    assert res.exit_code == 1


def test_yes_still_autoapproves_consent(repo, monkeypatch):
    fake = _fake_research(repo, {"type": "consent", "message": "问题中包含个人财务信息"})
    monkeypatch.setattr(cli_mod, "_services", lambda: SimpleNamespace(research=fake))
    CliRunner().invoke(app, ["research", "测试问题内容", "--yes"])
    assert fake.resumed == [True]


# ---------- #10：扫描取数失败可见 ----------

def test_scan_all_logs_fetch_failures(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    services = Services(svc.deps, svc)
    services.watch.store.subscribe(CODE, "示例酒业", 10)

    def boom(*a, **k):
        raise SkillFailed("数据源不可用")

    for name in ("_new_report", "_announcements", "_price_move"):
        setattr(services.watch, name, boom)
    logs: list[str] = []
    services.watch.scan_all(log=logs.append)
    assert any("取数失败" in ln for ln in logs)           # 不能把「全部取数失败」报成「没有变化」


# ---------- #11：对照评测自动批准 ----------

def test_compare_autoapprove_with_plain_research_service(repo):
    from advisor.evals.compare import run_compare

    class _CmpSvc:
        def __init__(self):
            self.deps = SimpleNamespace(repo=repo)
            self.resumed: list[bool] = []

        def start(self, req, background=False, namespace=None, **kw):
            repo.create_task("t_cmp", req.model_dump(mode="json") | {"namespace": namespace})
            repo.update_task("t_cmp", status="waiting_approval", interrupt={"type": "consent"})
            return "t_cmp"

        def resume(self, tid, approve, background=False, **kw):
            self.resumed.append(approve)
            repo.update_task(tid, status="failed", error="测试结束")

    svc = _CmpSvc()
    run_compare(svc, [{"id": "x", "question": f"复核 {CODE}", "ticker": CODE, "as_of": "2026-10-18"}], ["standard"])
    assert svc.resumed == [True]                         # svc 本身就是 ResearchService，自动批准不能调崩


# ---------- #12：无上一在用版本不可回滚 ----------

def test_strategy_rollback_without_previous_raises(repo, policy):
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    book = services.strategy
    v = book.create(spec(), "owner", "回滚无上一版本的基础测试记录")
    book.start_paper(v.card_id, v.version, "owner", start=START)
    run_days(book, START, 10)
    book.submit(v.card_id, v.version, "owner")
    book.approve(v.card_id, v.version, "owner")          # 首个晋级版本：没有前任
    with pytest.raises(StrategyError, match="没有上一个晋级"):
        book.rollback(v.card_id, "owner", "没有在用版本可恢复的回滚测试")


# ---------- #13：未来时点拒绝 ----------

def test_future_as_of_rejected():
    with pytest.raises(ValidationError):
        ResearchRequest(question="测试问题内容", as_of=date.today() + timedelta(days=30))
    ResearchRequest(question="测试问题内容", as_of=date.today())
