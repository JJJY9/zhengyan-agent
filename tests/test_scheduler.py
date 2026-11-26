"""后台定时器：间隔 0 不启动；一个子系统失败不拖累另一个；stop 能及时停止。"""
import threading

from advisor.scheduler import Scheduler


class FakeWatch:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def scan_all(self, log=None, is_cancelled=lambda: False):
        if self.fail:
            raise RuntimeError("扫描失败")
        return []


class FakeStrategy:
    def __init__(self, on_tick=None):
        self.called = 0
        self.on_tick = on_tick

    def step_all(self, day=None, log=None, is_cancelled=lambda: False):
        self.called += 1
        if self.on_tick:
            self.on_tick()
        return []


class FakeServices:
    def __init__(self, watch=None, strategy=None):
        self.watch = watch or FakeWatch()
        self.strategy = strategy or FakeStrategy()


def test_zero_interval_does_not_start_thread():
    s = Scheduler(FakeServices(), 0)
    s.start()
    assert not s._thread.is_alive()
    s.stop()


def test_tick_isolates_subsystem_failures():
    strategy = FakeStrategy()
    s = Scheduler(FakeServices(watch=FakeWatch(fail=True), strategy=strategy), 0)
    s.tick()                                # 扫描失败不应抛出，也不应跳过模拟盘推进
    assert strategy.called == 1


def test_loop_ticks_and_stops():
    fired = threading.Event()
    s = Scheduler(FakeServices(strategy=FakeStrategy(on_tick=fired.set)), interval_hours=0.0001)
    s.start()
    assert fired.wait(timeout=5), "定时器未在预期时间内触发"
    s.stop()
    s._thread.join(timeout=5)
    assert not s._thread.is_alive()
