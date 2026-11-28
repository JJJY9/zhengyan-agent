"""dossier CLI 回归：列出/查看/历史版本/缺失退出码——只读命令，不触发研究。"""
from types import SimpleNamespace

from typer.testing import CliRunner

import advisor.cli as cli_mod
from advisor.cli import app
from advisor.contracts import ResearchRequest
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service


def _run(app_args):
    return CliRunner().invoke(app, app_args)


def _seed(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩与估值", as_of=as_of(), route="quick"),
              background=False)
    return SimpleNamespace(deps=svc.deps)


def test_dossier_list_shows_current_versions(repo, policy, monkeypatch):
    services = _seed(repo, policy)
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res = _run(["dossier", "list"])
    assert res.exit_code == 0
    assert CODE in res.output and "v1" in res.output and "截至" in res.output
    res2 = _run(["dossier", "list", "--limit", "1"])
    assert res2.exit_code == 0


def test_dossier_show_current_and_history(repo, policy, monkeypatch):
    services = _seed(repo, policy)
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res = _run(["dossier", "show", CODE])
    assert res.exit_code == 0
    assert "质量门控" in res.output and "问题：" in res.output and "v1" in res.output
    res_v1 = _run(["dossier", "show", CODE, "--version", "1"])
    assert res_v1.exit_code == 0 and "v1" in res_v1.output


def test_dossier_missing_key_and_empty_list(repo, policy, monkeypatch):
    monkeypatch.setattr(cli_mod, "_services", lambda: SimpleNamespace(deps=SimpleNamespace(repo=repo)))
    res_empty = _run(["dossier", "list"])
    assert res_empty.exit_code == 0 and "还没有档案" in res_empty.output
    res = _run(["dossier", "show", "999999"])
    assert res.exit_code == 1 and "没有找到档案" in res.output
    res_v = _run(["dossier", "show", "999999", "--version", "3"])
    assert res_v.exit_code == 1 and "v3" in res_v.output
