"""health CLI 回归：正常自检输出与 Mongo 连接失败的退出码。"""
from types import SimpleNamespace

from typer.testing import CliRunner

import advisor.cli as cli_mod
from advisor.cli import app
from tests.test_phase2 import make_service


def _run(args):
    return CliRunner().invoke(app, args)


def test_health_reports_environment(repo, policy, monkeypatch):
    svc = make_service(repo, policy)
    monkeypatch.setattr(cli_mod, "_services", lambda: SimpleNamespace(deps=svc.deps))
    res = _run(["health"])
    assert res.exit_code == 0
    assert "MongoDB：连接正常" in res.output
    assert "技能：" in res.output and "大模型：" in res.output
    assert "RSI 在用版本：" in res.output and "数据量：" in res.output


def test_health_mongo_failure_exit_code(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    fake_db = SimpleNamespace(command=boom)
    deps = SimpleNamespace(repo=SimpleNamespace(db=fake_db))
    monkeypatch.setattr(cli_mod, "_services", lambda: SimpleNamespace(deps=deps))
    res = _run(["health"])
    assert res.exit_code == 1
    assert "连接失败" in res.output
