"""后台定时：按间隔扫描已订阅的股票、把模拟盘推进到今天。没有变化不产生提醒。"""
import logging
import threading
import time

from advisor.harness import TaskCancelled
from advisor.services import Services

log = logging.getLogger("advisor.scheduler")


class Scheduler:
    FAST_TICK_S = 60          # 分钟级轻量回收：僵尸任务/评测不依赖 6h 扫描节奏

    def __init__(self, services: Services, interval_hours: float):
        self.services = services
        self.interval_s = interval_hours * 3600
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="advisor-scheduler", daemon=True)

    def start(self) -> "Scheduler":
        if self.interval_s > 0:
            self._thread.start()
        return self

    def stop(self) -> None:
        """置停止位并等待当前 tick 在检查点退出：守护线程在净值/成交两次写之间被杀会留下缺失成交。"""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15)

    def tick(self) -> None:
        # 两个子系统各自兜底：订阅扫描失败不能拖累模拟盘推进（反之亦然）；停机时在检查点退出
        try:
            alerts = self.services.watch.scan_all(log=log.info, is_cancelled=self._stop.is_set)
            log.info("定时任务：新提醒 %d", len(alerts))
        except TaskCancelled:
            log.info("停机：订阅扫描在检查点停止")
            return
        except Exception:
            log.exception("订阅扫描失败")
        if self._stop.is_set():
            return
        try:
            points = self.services.strategy.step_all(log=log.info, is_cancelled=self._stop.is_set)
            log.info("定时任务：模拟盘推进 %d", len(points))
        except TaskCancelled:
            log.info("停机：模拟盘推进在检查点停止")
        except Exception:
            log.exception("模拟盘推进失败")

    def _loop(self) -> None:
        next_scan = time.monotonic() + self.interval_s
        while True:
            wait = min(self.FAST_TICK_S, max(next_scan - time.monotonic(), 0.0))
            if self._stop.wait(wait):
                return
            try:
                repo = self.services.deps.repo     # 回收是尽力而为：拿不到仓库也不能拖死定时线程
                repo.recover_stale_tasks(1800)      # 属主已死的任务最迟一个租约周期内回收
                repo.recover_stale_evals()
            except Exception:      # noqa: BLE001
                log.exception("僵尸回收失败")
            if time.monotonic() >= next_scan:
                try:
                    self.tick()
                except Exception:       # 定时任务失败不影响服务
                    log.exception("定时任务失败")
                next_scan = time.monotonic() + self.interval_s
