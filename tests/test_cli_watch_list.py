"""watch list CLI 回归：订阅/未关闭提醒展示与 --all/停用过滤——只读命令。"""
from datetime import datetime, timezone
from types import SimpleNamespace

from typer.testing import CliRunner

import advisor.cli as cli_mod
from advisor.cli import app
from advisor.services import Services
from advisor.watch.models import Alert
from tests.fakes import CODE, NAME
from tests.test_phase2 import make_service


def _run(args):
    return CliRunner().invoke(app, args)


def _alert(aid, status="open"):
    return Alert(id=aid, ticker=CODE, name=NAME, dossier_key=CODE, dossier_version=3,
                 fingerprint=f"fp_{aid}", events=[], affected=[], status=status,
                 created_at=datetime.now(timezone.utc))


def test_watch_list_shows_subs_and_open_alerts(repo, policy, monkeypatch):
    svc = make_service(repo, policy)
    services = SimpleNamespace(watch=Services(svc.deps, svc).watch)
    store = services.watch.store
    store.subscribe(CODE, NAME, 8)
    store.add_alert(_alert("al_open"))
    store.add_alert(_alert("al_dis"))
    store.update_alert("al_dis", status="dismissed")
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res = _run(["watch", "list"])
    assert res.exit_code == 0
    assert CODE in res.output and "阈值 ±8%" in res.output and "尚未扫描" in res.output
    assert "al_open" in res.output and "al_dis" not in res.output   # 已忽略的提醒不列出


def test_watch_list_all_includes_inactive(repo, policy, monkeypatch):
    svc = make_service(repo, policy)
    services = SimpleNamespace(watch=Services(svc.deps, svc).watch)
    store = services.watch.store
    store.subscribe(CODE, NAME)
    store.unsubscribe(CODE)
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res_active = _run(["watch", "list"])
    assert res_active.exit_code == 0 and "还没有" in res_active.output
    res_all = _run(["watch", "list", "--all"])
    assert res_all.exit_code == 0 and "已停用" in res_all.output
