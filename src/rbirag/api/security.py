"""API-key auth and a per-key sliding-window rate limiter (in-memory, single process)."""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque


def key_id(key: str) -> str:
    """Stable, non-reversible label for logs. The raw key is never logged."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class ApiKeys:
    def __init__(self, keys: str | list[str]):
        raw = keys.split(",") if isinstance(keys, str) else keys
        self._keys = [k.strip() for k in raw if k.strip()]

    def valid(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        ok = False
        for k in self._keys:  # compare against every key: constant time per comparison
            ok |= hmac.compare_digest(candidate.encode(), k.encode())
        return ok


class RateLimiter:
    """Allow `limit` requests per `window` seconds per identity."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = self.clock()
        with self._lock:
            q = self._hits[identity]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(0.0, self.window - (now - q[0]))
            q.append(now)
            return True, 0.0
