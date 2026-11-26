"""策略只从配置加载；运行中的模型无权修改。"""
import math
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from advisor.harness.errors import HarnessDenied
from advisor.settings import load_yaml

# 代码里直接 h.llm(role) 使用的角色（intake 识别、自动路由判别、起草），
# policies.yaml 未必引用它们，删改后只会运行期静默退化——启动期一并校验
CODE_REQUIRED_ROLES = ("classify", "draft", "reason")


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
        self._sites_raw = dict(sites)

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "Policy":
        policy = cls(load_yaml("policies.yaml", config_dir), load_yaml("sites.yaml", config_dir))
        roles: set[str] | None = None
        try:
            models = load_yaml("models.yaml", config_dir)
        except FileNotFoundError:
            models = None              # 合成配置（测试/嵌入）可以只给 policies+sites
        if models is not None:
            raw_roles = models.get("roles") if isinstance(models, dict) else None
            if not isinstance(raw_roles, dict) or not raw_roles:
                raise ValueError("models.yaml 的 roles 缺失/为空/不是映射：没有任何可用模型角色，拒绝启动")
            roles = set(raw_roles)
        policy.validate(roles)
        return policy

    def validate(self, models_roles: set[str] | None = None) -> None:
        """配置缺段/拼错必须启动即失败：egress 脱敏或抓取白名单静默失效比拒绝启动危险得多。"""
        unknown_sites = sorted(set(self._sites_raw) - {"fetch_allow", "block"})
        if unknown_sites:
            raise ValueError(f"sites.yaml 含未知键：{unknown_sites}（只允许 fetch_allow / block）")
        allow = self._sites_raw.get("fetch_allow")
        if not isinstance(allow, list) or not allow or not all(isinstance(s, str) and s.strip() for s in allow):
            raise ValueError("sites.yaml fetch_allow 缺失或含非字符串：抓取白名单会静默失效，拒绝启动")
        block = self._sites_raw.get("block")
        if block is not None and (not isinstance(block, list) or not all(isinstance(s, str) for s in block)):
            raise ValueError("sites.yaml block 必须是字符串列表")
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
        if models_roles is not None:
            missing_code = [r for r in CODE_REQUIRED_ROLES if r not in models_roles]
            if missing_code:
                raise ValueError(f"models.yaml roles 缺少代码硬编码角色：{missing_code}"
                                 "——入口识别/路由判别/起草会在运行时静默退化，拒绝启动")
            bad_roles: dict[str, str] = {}
            for name, rp in self.routes.items():
                if rp.llm_role not in models_roles:
                    bad_roles[f"routes.{name}.llm_role"] = rp.llm_role
            for key, role in (self.experts.get("roles") or {}).items():
                if role not in models_roles:
                    bad_roles[f"experts.roles.{key}"] = role
            if bad_roles:
                raise ValueError(f"models.yaml 未定义这些角色：{bad_roles}（已定义：{sorted(models_roles)}）"
                                 "——角色名拼错会在首个研究任务时才 KeyError，启动即拒绝加载")

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
