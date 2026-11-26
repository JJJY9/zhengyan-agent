"""策略只从配置加载；运行中的模型无权修改。"""
import math
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from advisor.harness.errors import HarnessDenied
from advisor.settings import load_yaml


@dataclass(frozen=True)
class RoutePolicy:
    name: str
    llm_role: str
    allowed_skills: tuple[str, ...]
    budget: dict[str, float]
    max_iterations: int


class Policy:
    def __init__(self, raw: dict[str, Any], sites: dict[str, Any]):
        self.routes = {
            name: RoutePolicy(
                name=name,
                llm_role=r["llm_role"],
                allowed_skills=tuple(r["allowed_skills"]),
                budget=dict(r["budget"]),
                max_iterations=int(r.get("max_iterations", 0)),
            )
            for name, r in raw["routes"].items()
        }
        self.approvals = raw.get("approvals", {})
        # 审批等级只允许 auto/confirm/deny：大小写或拼写偏差（Deny/Confirm/require）会被
        # _admit 当作未知级别放行，审批门静默失效——启动即拒绝加载
        levels = {"default": self.approvals.get("default", "deny"), **self.approvals.get("skills", {})}
        bad = sorted(k for k, v in levels.items() if v not in ("auto", "confirm", "deny"))
        if bad:
            raise ValueError(f"审批等级配置无效（只允许 auto/confirm/deny）：{bad}")
        self.egress = raw.get("egress", {})
        self.search = raw.get("search", {})
        self.routing = raw.get("routing", {})
        self.intake = raw.get("intake", {})
        self.experts = raw.get("experts", {})
        self.watch = raw.get("watch", {})
        self.strategy = raw.get("strategy", {})
        self.fetch_allow = tuple(sites.get("fetch_allow", []))
        self.block_sites = tuple(sites.get("block", []))

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "Policy":
        policy = cls(load_yaml("policies.yaml", config_dir), load_yaml("sites.yaml", config_dir))
        policy.validate()
        return policy

    def validate(self) -> None:
        """配置缺段/拼错必须启动即失败：egress 脱敏静默失效比拒绝启动危险得多。"""
        missing = [name for name in ("quick", "standard", "deep", "mcp") if name not in self.routes]
        if missing:
            raise ValueError(f"policies.yaml 缺少必需流程：{missing}")
        for name, rp in self.routes.items():
            for key in ("tokens", "search_calls", "fetch_calls", "wall_clock_s"):
                if key not in rp.budget:
                    raise ValueError(f"流程 {name} 缺少预算键 {key}")
                v = rp.budget[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    raise ValueError(f"流程 {name} 的预算 {key}={v!r} 无效（需为非负有限数）")
            if rp.budget["wall_clock_s"] <= 0:
                raise ValueError(f"流程 {name} 的 wall_clock_s 必须为正")
        pats = self.egress.get("sensitive_patterns")
        if not isinstance(pats, list) or not pats:
            raise ValueError("egress.sensitive_patterns 缺失或为空：出域脱敏会静默失效，拒绝启动")
        trade = self.egress.get("trade_intent_patterns")
        if not isinstance(trade, list):
            raise ValueError("egress.trade_intent_patterns 必须是列表")
        for pattern in [*pats, *trade]:
            try:
                re.compile(pattern)
            except (TypeError, re.error) as e:
                raise ValueError(f"egress 正则无效：{pattern!r}（{e}）") from e

    def route(self, name: str) -> RoutePolicy:
        if name not in self.routes:
            raise HarnessDenied(f"未知流程: {name}")
        return self.routes[name]

    def is_allowed(self, route: str, skill: str) -> bool:
        return any(fnmatch(skill, pattern) for pattern in self.route(route).allowed_skills)

    def approval_level(self, skill: str) -> str:
        for pattern, level in self.approvals.get("skills", {}).items():
            if fnmatch(skill, pattern):
                return level
        return self.approvals.get("default", "deny")
