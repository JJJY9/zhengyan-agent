import threading
import time


class TokenBucket:
    """账号级限速（DeepSeek Search/Fetch 各 5 QPS），进程内共享。"""

    def __init__(self, rate: float, capacity: float | None = None):
        if rate <= 0:
            raise ValueError(f"限速必须为正数（qps={rate}）")
        self.rate = float(rate)
        # 容量至少 1：rate<1 时若按 rate 当容量，令牌永远补不满 1 个，acquire 会无限等待
        self.capacity = max(1.0, float(capacity if capacity is not None else rate))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            time.sleep(wait)
