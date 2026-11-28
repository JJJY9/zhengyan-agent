"""FastMCP：把只读 Skill 暴露给外部 Agent。调用同样经过 Harness（mcp 流程白名单与预算）。"""
from datetime import date

from fastmcp import FastMCP

from advisor.contracts.dossier import Dossier
from advisor.evidence import sanitize
from advisor.graph.deps import Deps, build_default_deps
from advisor.harness import Budget


def _dossier_snapshot(d: Dossier) -> dict:
    """外部 Agent 需要的档案要点（不含报告全文与内部编排细节）。"""
    return {
        "key": d.key, "version": d.version, "ticker": d.ticker, "name": d.name,
        "as_of": d.as_of.isoformat(), "status": d.status, "summary": d.summary,
        "quality": d.quality.model_dump(mode="json"),
        "claims": [{"kind": c.kind, "status": c.status, "text": c.text} for c in d.claims],
        "unknowns": list(d.unknowns),
        "dissents": [x.model_dump(mode="json") for x in d.dissents],
        "review_triggers": [x.model_dump(mode="json") for x in d.review_triggers],
        "prior_review": [x.model_dump(mode="json") for x in d.prior_review],
        "boundary_notes": list(d.boundary_notes),
        "evidence_ids": list(d.evidence_ids),
    }


def tool_functions(deps: Deps) -> dict:
    """只读工具的实现（数据源技能 + 档案读取），MCP 注册与单元测试共用。"""

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
        out = call("web.search", query=query, as_of=as_of)
        # 外部 Agent 直接消费原始结果：与内部 gather 同口径清洗标题/摘要/片段，
        # 注入文本（<script>、「忽略之前的指令」等）不能经 MCP 出口绕过防护
        for hit in out.get("hits", []):
            hit["title"] = sanitize(hit.get("title") or "", max_chars=200)[0]
            hit["summary"] = sanitize(hit.get("summary") or "", max_chars=800)[0]
            hit["snippet"] = sanitize(hit.get("snippet") or "", max_chars=800)[0]
        return out

    def dossier_current(ticker: str, as_of: date | None = None) -> dict:
        """某标的「当前档案」快照（按研究时点取最新版本）：结论摘要、主张核验状态、
        未决问题、复核触发条件与质量门控。只读，不含报告全文。"""
        key = (ticker or "").strip()
        if not key:
            raise ValueError("ticker 不能为空")
        if ":" in key:
            raise ValueError("评测等命名空间档案不对 MCP 开放")
        d = deps.repo.current_dossier(key, as_of or date.today())
        if d is None:
            return {"found": False, "ticker": key, "message": f"还没有 {key} 的档案"}
        return {"found": True, **_dossier_snapshot(d)}

    def dossier_list(limit: int = 20) -> dict:
        """已跟踪标的的当前档案概要列表（每个标的取研究时点最新的版本）。"""
        rows = deps.repo.list_dossiers()[: max(1, min(int(limit), 100))]
        return {"count": len(rows), "dossiers": [
            {"key": r["key"], "ticker": r.get("ticker", ""), "name": r.get("name", ""),
             "version": r["version"], "as_of": str(r.get("as_of") or "")[:10],
             "status": r.get("status", ""), "summary": r.get("summary", "")}
            for r in rows
        ]}

    return {"financial_statements": financial_statements, "market_daily": market_daily,
            "announcements": announcements, "web_search": web_search,
            "dossier_current": dossier_current, "dossier_list": dossier_list}


def build_server(deps: Deps | None = None) -> FastMCP:
    deps = deps or build_default_deps()
    mcp = FastMCP("stock-advisor-skills")
    for fn in tool_functions(deps).values():
        mcp.tool(fn)
    return mcp
