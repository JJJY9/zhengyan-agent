"""MCP 只读工具：经 Harness（mcp 流程白名单与预算）执行；出域检查生效。"""
import asyncio

import pytest

from advisor.harness import HarnessDenied
from advisor.mcp_server import build_server, tool_functions
from tests.fakes import CODE, as_of


def test_tools_return_skill_cards(service):
    fns = tool_functions(service.deps)
    quote = fns["market_daily"](CODE, as_of())
    assert quote["cards"] and quote["cards"][0]["level"] == "A"
    filing = fns["financial_statements"](CODE, as_of())
    assert filing["cards"] and filing["cards"][0]["skill"] == "financials.statements"
    assert fns["announcements"](CODE, as_of())["cards"]


def test_web_search_blocks_sensitive_query(service):
    fns = tool_functions(service.deps)
    with pytest.raises(HarnessDenied, match="敏感"):
        fns["web_search"]("我的持仓 30万元 " + CODE, as_of())
    out = fns["web_search"]("示例酒业 业绩", as_of())
    assert "hits" in out


def test_build_server_registers_tools(service):
    mcp = build_server(service.deps)
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == {
        "financial_statements", "market_daily", "announcements", "web_search",
        "dossier_current", "dossier_list",
    }
