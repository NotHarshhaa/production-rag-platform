"""Thread-safe sliding-window rate limiting shared by the IP limiter and
per-API-key quotas.

Each limiter instance tracks one limit; callers keep a map of limit -> key
state so limits can differ per client (see app/api/auth.py).
"""

import threading
import time
from collections import deque


class SlidingWindowLimiter:
    """Sliding-window counter: at most ``limit`` events per ``window_seconds``.

    One instance can track many subjects (IPs or API keys); each subject gets
    its own event deque. ``limit <= 0`` disables limiting.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, subject: str) -> bool:
        if self.limit <= 0:
            return True
        window_start = time.monotonic() - self.window_seconds
        with self._lock:
            hits = self._hits.setdefault(
                subject, deque(maxlen=max(1, self.limit))
            )
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(time.monotonic())
            return True

    def clear(self, subject: str | None = None) -> None:
        with self._lock:
            if subject is None:
                self._hits.clear()
            else:
                self._hits.pop(subject, None)
