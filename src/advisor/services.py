"""应用服务容器：API 与命令行共用同一套组装方式。

research  研究任务（主图、人在环、取消）
rsi       受控自进化
profiles  用户确认档案（只有用户确认这一条写入路径）
watch     事件订阅与增量复核
strategy  策略卡与前向模拟盘
"""
from dataclasses import dataclass, field

from advisor.graph.deps import Deps
from advisor.graph.runner import ResearchService
from advisor.personal import ProfileStore
from advisor.rsi import RSIRunner
from advisor.strategy import StrategyBook, StrategyStore
from advisor.watch import Watcher, WatchStore

STARTUP_RECOVER_AGE_S = 600       # 启动即回收：一次模型调用最长 300s，停滞 10 分钟的任务线程不可能还活着


@dataclass
class Services:
    deps: Deps
    research: ResearchService
    # 只有服务进程（uvicorn）启动时才回收僵尸任务：CLI 每次命令都会构造 Services，
    # 若无条件回收会把其它进程正在跑的研究任务标成失败
    recover_stale: bool = False
    rsi: RSIRunner = field(init=False)
    profiles: ProfileStore = field(init=False)
    watch: Watcher = field(init=False)
    strategy: StrategyBook = field(init=False)

    def __post_init__(self):
        db = self.deps.repo.db
        if self.recover_stale:
            # 服务进程重启后把上一实例遗留的 queued/running 任务标记失败（重启前的任务线程已不存在）
            self.deps.repo.recover_stale_tasks(STARTUP_RECOVER_AGE_S)
            self.deps.repo.recover_stale_evals()          # 评测同理：running 僵尸不再让页面永远轮询
        self.rsi = RSIRunner(self.deps, self.deps.ledger, self.deps.rsi)
        self.profiles = ProfileStore(db)
        self.watch = Watcher(self.deps, WatchStore(db))
        self.strategy = StrategyBook(self.deps, StrategyStore(db), open_alerts=self.open_alert_count)

    def confirm_profile(self, data, actor: str, confirmed: bool):
        """保存新版本约束，并标出哪些档案的个人情境结论因此改变（个人信息只在本地比对）。"""
        from advisor.personal import profile_impact
        v = self.profiles.confirm(data, actor, confirmed)
        prev = self.profiles.get(v.parent_version)   # 并发确认时 v.parent_version 可能已前进：按真实父版本算影响
        repo = self.deps.repo
        impact = []
        for row in repo.list_dossiers():
            d = repo.current_dossier(row["key"])      # 当前档案按研究时点取，回填的高版本旧档案不算最新
            if d is None:
                continue
            sub = self.watch.store.get_sub(d.ticker)
            alerts = [(a.id, "、".join(e.title for e in a.events))
                      for a in self.watch.store.list_alerts(d.ticker) if a.status in ("open", "reviewing")]
            if hit := profile_impact(d, repo.get_evidence(d.evidence_ids), prev, v, bool(sub and sub.active), alerts,
                                     sub.last_scan_note if sub else "", sub.last_scan_at if sub else None):
                impact.append(hit)
        self.profiles.set_impact(v.version, impact)
        v.impact = impact
        return v

    def open_alert_count(self, ticker: str) -> int:
        return len(self.watch.store.list_alerts(ticker, "open"))

    def submit(self, fn, *args):
        # 页面上的后台作业（RSI、评测、扫描、模拟盘推进）走侧栏线程池：不占交互研究的名额
        return self.research.submit_side(fn, *args)


def build_services(deps: Deps | None = None, checkpointer=None, recover_stale: bool = False) -> Services:
    if deps is None:
        from langgraph.checkpoint.mongodb import MongoDBSaver

        from advisor.graph.deps import build_default_deps
        deps = build_default_deps()
        checkpointer = MongoDBSaver(deps.repo.db.client, db_name=f"{deps.repo.db.name}_checkpoints")
    return Services(deps, ResearchService(deps, checkpointer), recover_stale=recover_stale)
