"""策略只从配置加载；运行中的模型无权修改。"""
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
        return cls(load_yaml("policies.yaml", config_dir), load_yaml("sites.yaml", config_dir))

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
