"""FastAPI service: /health, /ask, /ask/stream (SSE). API key + per-key rate limit.

Run:  uvicorn rbirag.api.app:create_app --factory --port 8000
Single worker on purpose: the embedded Qdrant index is opened by one process.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..generation.answer import RAG, Answer
from ..generation.llm import BudgetExceeded
from ..observability.tracing import Trace, TraceStore
from .security import ApiKeys, RateLimiter, key_id

log = logging.getLogger("rbirag.api")
VERSION = "0.1.0"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="A question about RBI regulations")


def answer_to_dict(a: Answer, request_id: str, latency_ms: float) -> dict:
    return {
        "request_id": request_id,
        "question": a.question,  # PII-masked form only
        "answer": a.answer,
        "abstained": a.abstained,
        "reason": a.reason,
        "citations": [c.__dict__ for c in a.citations],
        "guardrails": {"pii_masked": a.pii_types, "injection_detected": a.injection_detected,
                       "excerpts_dropped": a.dropped_hits},
        "usage": ({"model": a.llm.model, "input_tokens": a.llm.input_tokens,
                   "output_tokens": a.llm.output_tokens, "cost_usd": round(a.llm.cost_usd, 6),
                   "cached": a.llm.cached} if a.llm else None),
        "latency_ms": round(latency_ms, 1),
    }


def create_app(rag: RAG | None = None, api_keys: str | None = None,
               rate_limit_per_min: int | None = None, admin_keys: str | None = None,
               trace_store: TraceStore | None = None) -> FastAPI:
    keys = ApiKeys(api_keys if api_keys is not None else settings.api_keys)
    admins = ApiKeys(admin_keys if admin_keys is not None else settings.admin_api_keys)
    store = trace_store or TraceStore(settings.trace_dir / "traces.jsonl")
    limiter = RateLimiter(rate_limit_per_min or settings.rate_limit_per_min)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = rag or RAG()  # loads embedding models + opens the index once at startup
        yield

    app = FastAPI(title="RBI Regulation RAG", version=VERSION, lifespan=lifespan,
                  description="Grounded answers with page-level citations over public RBI documents.")

    def auth(x_api_key: str | None = Header(default=None)) -> str:
        if not keys.valid(x_api_key):
            raise HTTPException(401, "Missing or invalid X-API-Key",
                                headers={"WWW-Authenticate": "ApiKey"})
        return x_api_key

    def limit(response: Response, key: str = Depends(auth)) -> str:
        ok, retry = limiter.check(key)
        if not ok:
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)})
        return key

    def admin(x_api_key: str | None = Header(default=None)) -> str:
        if not admins._keys:
            raise HTTPException(403, "Trace endpoints are disabled: no admin keys configured")
        if not admins.valid(x_api_key):
            raise HTTPException(401, "Admin key required", headers={"WWW-Authenticate": "ApiKey"})
        return x_api_key

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        log.info(json.dumps({"event": "http", "rid": rid, "path": request.url.path,
                             "status": resp.status_code,
                             "ms": round((time.perf_counter() - t0) * 1000, 1)}))
        return resp

    @app.exception_handler(BudgetExceeded)
    async def _budget(request: Request, exc: BudgetExceeded):
        return JSONResponse({"detail": "Demo budget exhausted. Please try again later."}, status_code=503)

    @app.get("/health", tags=["ops"])
    def health():
        r = app.state.rag.retriever
        try:
            n = r.client.count(r.collection).count
        except Exception:  # fakes / unavailable index
            n = None
        return {"status": "ok" if n else "degraded", "version": VERSION, "chunks_indexed": n,
                "gen_model": settings.gen_model}

    @app.post("/ask", tags=["qa"])
    def ask(body: AskRequest, request: Request, key: str = Depends(limit)):
        t0 = time.perf_counter()
        trace = Trace(request.state.request_id, "ask", key_id(key))
        try:
            a = app.state.rag.answer(body.question, trace=trace)
        except Exception as e:
            store.add(trace.finish(error=e))
            raise
        store.add(trace.finish(a))
        ms = (time.perf_counter() - t0) * 1000
        log.info(json.dumps({"event": "ask", "rid": request.state.request_id, "key": key_id(key),
                             "reason": a.reason, "abstained": a.abstained,
                             "pii": list(a.pii_types), "injection": a.injection_detected,
                             "ms": round(ms, 1)}))
        return answer_to_dict(a, request.state.request_id, ms)

    @app.post("/ask/stream", tags=["qa"])
    def ask_stream(body: AskRequest, request: Request, key: str = Depends(limit)):
        rid, t0 = request.state.request_id, time.perf_counter()
        trace = Trace(rid, "ask_stream", key_id(key))

        def sse(event: str, data) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def gen():
            try:
                for ev in app.state.rag.stream(body.question, trace=trace):
                    if ev["event"] == "final":
                        ms = (time.perf_counter() - t0) * 1000
                        a = ev["data"]
                        store.add(trace.finish(a))
                        log.info(json.dumps({"event": "ask_stream", "rid": rid, "key": key_id(key),
                                             "reason": a.reason, "ms": round(ms, 1)}))
                        yield sse("final", answer_to_dict(a, rid, ms))
                    else:
                        yield sse(ev["event"], ev["data"])
            except BudgetExceeded as e:
                store.add(trace.finish(error=e))
                yield sse("error", {"detail": "Demo budget exhausted. Please try again later."})
            except Exception as e:
                store.add(trace.finish(error=e))
                log.exception("stream failed rid=%s", rid)
                yield sse("error", {"detail": "Internal error"})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- observability (admin only: traces contain masked questions of all users) --------
    @app.get("/traces/summary", tags=["observability"])
    def traces_summary(_: str = Depends(admin)):
        return store.summary()

    @app.get("/traces", tags=["observability"])
    def traces_recent(limit: int = 20, _: str = Depends(admin)):
        return store.recent(max(1, min(limit, 200)))

    @app.get("/traces/{request_id}", tags=["observability"])
    def trace_by_id(request_id: str, _: str = Depends(admin)):
        t = store.get(request_id)
        if t is None:
            raise HTTPException(404, "Trace not found (only the most recent 500 are kept in memory)")
        return t

    return app
