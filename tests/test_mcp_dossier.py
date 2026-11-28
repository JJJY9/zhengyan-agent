"""MCP 档案读取工具：当前档案快照/列表/缺失/命名空间防护——只读，经仓库时点语义取当前版本。"""
from datetime import date

import pytest

from advisor.contracts import ResearchRequest
from advisor.mcp_server import tool_functions
from tests.fakes import CODE, NAME, as_of
from tests.test_phase2 import make_service


def _seed(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩与估值", as_of=as_of(), route="quick"),
              background=False)
    return svc.deps


def test_dossier_current_returns_snapshot(repo, policy):
    deps = _seed(repo, policy)
    out = tool_functions(deps)["dossier_current"](CODE)
    assert out["found"] is True
    assert out["version"] == 1 and out["ticker"] == CODE and out["name"] == NAME
    assert out["as_of"] == as_of().isoformat()
    assert out["quality"]["decision"] in ("pass", "need_more", "partial")
    assert out["claims"] and out["claims"][0]["status"] in (
        "candidate", "verified", "disputed", "rejected", "stale")
    # 快照只给要点：报告全文与原始问题不对外
    assert "report_md" not in out and "question" not in out
    assert "routing" not in out and "deliberation" not in out


def test_dossier_current_missing_and_namespaced(repo, policy):
    deps = _seed(repo, policy)
    fns = tool_functions(deps)
    out = fns["dossier_current"]("999999")
    assert out["found"] is False and "999999" in out["message"]
    with pytest.raises(ValueError, match="命名空间"):
        fns["dossier_current"]("eval-ev_1:600519")
    with pytest.raises(ValueError, match="不能为空"):
        fns["dossier_current"]("  ")


def test_dossier_list_and_as_of_window(repo, policy):
    deps = _seed(repo, policy)
    fns = tool_functions(deps)
    out = fns["dossier_list"]()
    assert out["count"] == 1
    row = out["dossiers"][0]
    assert row["key"] == CODE and row["version"] == 1 and row["as_of"] == as_of().isoformat()
    assert row["summary"] is not None
    # 时点在档案生成之前 → 该时点还没有档案（与研究时点语义一致）
    assert fns["dossier_current"](CODE, as_of=date(2000, 1, 1))["found"] is False
    # limit：默认给前 N 份，非法下限收敛为 1
    assert fns["dossier_list"](limit=0)["count"] == 1
    assert fns["dossier_list"](limit=500)["count"] == 1
