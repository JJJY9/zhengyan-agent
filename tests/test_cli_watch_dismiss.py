"""watch dismiss CLI 回归：忽略待处理提醒、状态校验与缺失退出码。"""
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


def _services_with_alert(repo, policy, aid="al_d1", status="open"):
    svc = make_service(repo, policy)
    services = SimpleNamespace(watch=Services(svc.deps, svc).watch)
    store = services.watch.store
    store.subscribe(CODE, NAME)
    store.add_alert(Alert(id=aid, ticker=CODE, name=NAME, dossier_key=CODE, dossier_version=2,
                          fingerprint=f"fp_{aid}", events=[], affected=[], status=status,
                          created_at=datetime.now(timezone.utc)))
    return services


def test_watch_dismiss_marks_dismissed(repo, policy, monkeypatch):
    services = _services_with_alert(repo, policy)
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res = _run(["watch", "dismiss", "al_d1", "--note", "已知晓"])
    assert res.exit_code == 0 and "已忽略" in res.output
    alert = services.watch.store.get_alert("al_d1")
    assert alert.status == "dismissed" and alert.note == "已知晓"


def test_watch_dismiss_rejects_non_open_and_missing(repo, policy, monkeypatch):
    services = _services_with_alert(repo, policy, aid="al_d2", status="resolved")
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res = _run(["watch", "dismiss", "al_d2"])
    assert res.exit_code == 1 and "只有待处理" in res.output
    res_missing = _run(["watch", "dismiss", "al_none"])
    assert res_missing.exit_code == 1 and "提醒不存在" in res_missing.output
