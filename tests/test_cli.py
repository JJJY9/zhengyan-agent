"""CLI 冒烟：用假模型与 mongomock 组装的服务跑主要子命令（不联网、不需要 Key）。"""
from typer.testing import CliRunner

import advisor.cli as cli
from advisor.services import Services


def run(service, *args):
    svc = Services(service.deps, service)
    orig = cli._services
    cli._services = lambda: svc
    try:
        return CliRunner().invoke(cli.app, list(args))
    finally:
        cli._services = orig


def test_help_lists_modules():
    r = CliRunner().invoke(cli.app, ["--help"])
    assert r.exit_code == 0 and "rsi" in r.output and "eval" in r.output


def test_rsi_collect_command(service):
    r = run(service, "rsi", "collect")
    assert r.exit_code == 0, r.output
    assert "新增坏案例" in r.output


def test_eval_routing_command(service):
    r = run(service, "eval", "routing")
    assert r.exit_code == 0, r.output
    assert "准确率" in r.output
