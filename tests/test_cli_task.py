"""task CLI 回归：列表/状态过滤/详情/缺失退出码——只读命令，不触发研究。"""
from types import SimpleNamespace

from typer.testing import CliRunner

import advisor.cli as cli_mod
from advisor.cli import app
from advisor.contracts import ResearchRequest
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service


def _run(args):
    return CliRunner().invoke(app, args)


def _seed(repo, policy):
    svc = make_service(repo, policy)
    tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩与估值", as_of=as_of(), route="quick"),
                    background=False)
    repo.create_task("t_failed_x", {"question": "复核 600999 业绩"})
    repo.update_task("t_failed_x", status="failed", error="模拟失败：取数超时")
    return tid, SimpleNamespace(deps=svc.deps)


def test_task_list_and_status_filter(repo, policy, monkeypatch):
    tid, services = _seed(repo, policy)
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res = _run(["task", "list"])
    assert res.exit_code == 0
    assert tid in res.output or "复核" in res.output
    res_f = _run(["task", "list", "--status", "failed"])
    assert res_f.exit_code == 0
    res_none = _run(["task", "list", "--status", "queued"])
    assert res_none.exit_code == 0 and "没有匹配的任务" in res_none.output


def test_task_show_details_and_missing(repo, policy, monkeypatch):
    tid, services = _seed(repo, policy)
    monkeypatch.setattr(cli_mod, "_services", lambda: services)
    res = _run(["task", "show", tid])
    assert res.exit_code == 0
    assert "问题：" in res.output and "档案：" in res.output
    res_missing = _run(["task", "show", "t_nope"])
    assert res_missing.exit_code == 1 and "没有找到任务" in res_missing.output
    # 失败任务的错误可见
    repo.create_task("t_failed_y", {"question": "复核 600999 业绩"})
    repo.update_task("t_failed_y", status="failed", error="模拟失败：取数超时")
    res_err = _run(["task", "show", "t_failed_y"])
    assert res_err.exit_code == 0 and "模拟失败" in res_err.output
