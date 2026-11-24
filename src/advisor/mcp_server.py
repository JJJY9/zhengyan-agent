"""FastMCP：把只读 Skill 暴露给外部 Agent。调用同样经过 Harness（mcp 流程白名单与预算）。"""
from datetime import date

from fastmcp import FastMCP

from advisor.graph.deps import Deps, build_default_deps
from advisor.harness import Budget


def tool_functions(deps: Deps) -> dict:
    """四个只读工具的实现，MCP 注册与单元测试共用。"""

    def call(skill: str, **args):
        as_of = args.get("as_of") or date.today()
        budget = Budget(limits=deps.policy.route("mcp").budget)
        h = deps.harness("mcp", "mcp", as_of, budget)
        return h.call(skill, **args).model_dump(mode="json")

    def financial_statements(code: str, as_of: date) -> dict:
        """as_of 时点已披露的 A 股利润表/现金流量表（A 级证据卡）。"""
        return call("financials.statements", code=code, as_of=as_of)

    def market_daily(code: str, as_of: date) -> dict:
        """as_of 前的日线行情统计。"""
        return call("market.daily", code=code, as_of=as_of)

    def announcements(code: str, as_of: date) -> dict:
        """as_of 前的巨潮公告标题。"""
        return call("announcements.list", code=code, as_of=as_of)

    def web_search(query: str, as_of: date) -> dict:
        """时点约束的网页搜索（结果只是线索）。"""
        return call("web.search", query=query, as_of=as_of)

    return {"financial_statements": financial_statements, "market_daily": market_daily,
            "announcements": announcements, "web_search": web_search}


def build_server(deps: Deps | None = None) -> FastMCP:
    deps = deps or build_default_deps()
    mcp = FastMCP("stock-advisor-skills")
    for fn in tool_functions(deps).values():
        mcp.tool(fn)
    return mcp
