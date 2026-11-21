"""Request tracing: one Trace per request, made of timed spans with attributes.

Passed explicitly (not via contextvars): streamed responses run each step on a different
thread, which would silently drop a context-based trace.

Privacy: spans only ever receive already-masked text and PII *types*. Stored traces are
therefore safe to keep and to expose to an admin.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path


class Span:
    __slots__ = ("name", "start_ms", "ms", "attrs", "error")

    def __init__(self, name: str, start_ms: float, attrs: dict):
        self.name, self.start_ms, self.ms, self.attrs, self.error = name, start_ms, 0.0, dict(attrs), None

    def set(self, **kw) -> None:
        self.attrs.update(kw)

    def to_dict(self) -> dict:
        d = {"name": self.name, "start_ms": round(self.start_ms, 1), "ms": round(self.ms, 1),
             **self.attrs}
        if self.error:
            d["error"] = self.error
        return d


class Trace:
    def __init__(self, request_id: str = "", endpoint: str = "", key: str = "", enabled: bool = True):
        self.request_id, self.endpoint, self.key, self.enabled = request_id, endpoint, key, enabled
        self.started_at, self._t0 = time.time(), time.perf_counter()
        self.spans: list[Span] = []
        self.outcome: dict = {}
        self.error: str | None = None
        self.total_ms = 0.0

    @contextmanager
    def span(self, name: str, **attrs):
        sp = Span(name, (time.perf_counter() - self._t0) * 1000, attrs)
        t = time.perf_counter()
        try:
            yield sp
        except Exception as e:  # record where it failed, then let it propagate
            sp.error = type(e).__name__
            raise
        finally:
            sp.ms = (time.perf_counter() - t) * 1000
            if self.enabled:
                self.spans.append(sp)

    def finish(self, answer=None, error: Exception | None = None) -> "Trace":
        self.total_ms = (time.perf_counter() - self._t0) * 1000
        if error is not None:
            self.error = type(error).__name__
        if answer is not None:
            llm = answer.llm
            self.outcome = {
                "reason": answer.reason, "abstained": answer.abstained,
                "answer": answer.answer[:400], "citations": [
                    {"n": c.n, "doc_id": c.doc_id, "para": c.section_no, "pages": c.pages}
                    for c in answer.citations],
                "pii_masked": answer.pii_types, "injection_detected": answer.injection_detected,
                "input_tokens": llm.input_tokens if llm else 0,
                "output_tokens": llm.output_tokens if llm else 0,
                "cost_usd": 0.0 if (llm is None or llm.cached) else round(llm.cost_usd, 6),  # actual spend
                "cached": llm.cached if llm else False,
            }
        return self

    def to_dict(self) -> dict:
        return {"request_id": self.request_id, "endpoint": self.endpoint, "key": self.key,
                "ts": round(self.started_at, 3), "total_ms": round(self.total_ms, 1),
                "error": self.error, "outcome": self.outcome,
                "spans": [s.to_dict() for s in self.spans]}


NULL_TRACE = Trace(enabled=False)  # shared no-op trace: lets pipeline code stay unconditional


class TraceStore:
    """Ring buffer in memory (fast lookups) + append-only JSONL file with size rotation."""

    def __init__(self, path: Path | None = None, ring: int = 500, max_bytes: int = 5_000_000):
        self.path, self.max_bytes = path, max_bytes
        self._ring: deque[dict] = deque(maxlen=ring)
        self._lock = threading.Lock()

    def add(self, trace: Trace) -> dict:
        d = trace.to_dict()
        with self._lock:
            self._ring.append(d)
            if self.path:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                        self.path.replace(self.path.with_suffix(".jsonl.1"))
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                except OSError:  # tracing must never take the service down
                    pass
        return d

    def get(self, request_id: str) -> dict | None:
        with self._lock:
            return next((t for t in reversed(self._ring) if t["request_id"] == request_id), None)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._ring)[-limit:][::-1]

    def summary(self) -> dict:
        with self._lock:
            ts = list(self._ring)
        if not ts:
            return {"requests": 0}
        lat = sorted(t["total_ms"] for t in ts)
        outs = [t["outcome"] for t in ts if t["outcome"]]
        pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
        return {
            "requests": len(ts), "errors": sum(1 for t in ts if t["error"]),
            "latency_ms": {"p50": round(statistics.median(lat), 1), "p95": round(pct(0.95), 1),
                           "max": round(lat[-1], 1)},
            "reasons": dict(Counter(o["reason"] for o in outs)),
            "abstain_rate": round(sum(o["abstained"] for o in outs) / max(1, len(outs)), 3),
            "pii_requests": sum(1 for o in outs if o["pii_masked"]),
            "injection_requests": sum(1 for o in outs if o["injection_detected"]),
            "tokens": {"input": sum(o["input_tokens"] for o in outs),
                       "output": sum(o["output_tokens"] for o in outs)},
            "cost_usd": round(sum(o["cost_usd"] for o in outs), 5),
            "cache_hit_rate": round(sum(o["cached"] for o in outs) / max(1, len(outs)), 3),
        }
