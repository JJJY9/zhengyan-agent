"""预算对象随图状态持久化，断点恢复后继续计数。"""
import time
from dataclasses import asdict, dataclass, field

from advisor.harness.errors import BudgetExceeded


@dataclass
class Budget:
    limits: dict[str, float]
    used: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    paused_s: float = 0.0            # 审批/停机等待累计秒数：不计入执行时限

    def elapsed(self) -> float:
        """实际执行时长：扣除等待审批的暂停时间。"""
        return max(time.time() - self.started_at - self.paused_s, 0.0)

    def wall_left(self) -> float:
        if "wall_clock_s" not in self.limits:
            return float("inf")
        return self.limits["wall_clock_s"] - self.elapsed()

    def remaining(self, kind: str) -> float:
        return self.limits.get(kind, 0) - self.used.get(kind, 0)

    def check(self, kind: str | None = None, amount: float = 1) -> None:
        elapsed = self.elapsed()
        if elapsed > self.limits.get("wall_clock_s", float("inf")):
            raise BudgetExceeded(f"总时限耗尽（{elapsed:.0f}s）")
        if kind and self.remaining(kind) < amount:
            raise BudgetExceeded(f"{kind} 预算耗尽（上限 {self.limits.get(kind, 0):g}）")

    def charge(self, kind: str, amount: float = 1) -> None:
        self.used[kind] = self.used.get(kind, 0) + amount

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Budget":
        return cls(limits=dict(data["limits"]), used=dict(data.get("used", {})), started_at=data["started_at"],
                   paused_s=float(data.get("paused_s", 0.0)))
