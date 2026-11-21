import json

import pytest
from fastapi.testclient import TestClient
from test_generation import FakeRetriever

from rbirag.api.app import create_app
from rbirag.generation.answer import RAG
from rbirag.generation.llm import LLMResult
from rbirag.observability.tracing import Trace, TraceStore


class StreamLLM:
    """Fake LLM with both complete() and stream()."""

    def __init__(self, text="Not less than one day [1]."):
        self.text = text

    def _res(self):
        return LLMResult(text=self.text, input_tokens=1000, output_tokens=40, cost_usd=0.0003,
                         model="claude-haiku-5-5")

    def complete(self, **kw):
        return self._res()

    def stream(self, **kw):
        outer = self

        class S:
            result = None

            def __iter__(s):
                for i in range(0, len(outer.text), 8):
                    yield outer.text[i:i + 8]
                s.result = outer._res()
        return S()


def make_rag(text="Not less than one day [1]."):
    return RAG(retriever=FakeRetriever(0.85), llm=StreamLLM(text), gate=0.67)


def names(t):
    return [s["name"] for s in t["spans"]]


def test_answer_records_every_pipeline_stage():
    tr = Trace("r1", "ask", "k")
    a = make_rag().answer("What is the cooling-off period?", trace=tr)
    tr.finish(a)
    d = tr.to_dict()
    assert names(d) == ["guardrails_input", "gate", "retrieval", "guardrails_excerpts", "llm",
                        "verification"]
    sp = {s["name"]: s for s in d["spans"]}
    assert sp["gate"]["passed"] is True and sp["gate"]["top_dense_score"] == 0.85
    assert len(sp["retrieval"]["chunks"]) == 3 and sp["retrieval"]["chunks"][0]["doc_id"] == "dl25"
    assert sp["llm"]["input_tokens"] == 1000 and sp["llm"]["cost_usd"] == 0.0003
    assert d["outcome"]["reason"] == "answered" and d["outcome"]["citations"][0]["para"] == "1"
    assert d["total_ms"] > 0 and all(s["ms"] >= 0 for s in d["spans"])


def test_out_of_domain_trace_stops_before_llm():
    tr = Trace("r2", "ask")
    RAG(retriever=FakeRetriever(0.4), llm=StreamLLM(), gate=0.67).answer("poem?", trace=tr)
    d = tr.to_dict()
    assert names(d) == ["guardrails_input", "gate"] and d["spans"][1]["passed"] is False


def test_stream_trace_has_first_token_latency():
    tr = Trace("r3", "ask_stream")
    events = list(make_rag().stream("cooling-off period?", trace=tr))
    tr.finish(events[-1]["data"])
    llm = next(s for s in tr.to_dict()["spans"] if s["name"] == "llm")
    assert llm["streamed"] is True and llm["token_events"] >= 3 and llm["first_token_ms"] >= 0


def test_trace_never_contains_raw_pii():
    tr = Trace("r4", "ask")
    a = make_rag().answer("My PAN is ABCDE1234F and Aadhaar 2345 6789 0123, can it be disbursed?",
                          trace=tr)
    blob = json.dumps(tr.finish(a).to_dict())
    assert "ABCDE1234F" not in blob and "2345 6789" not in blob
    assert tr.to_dict()["outcome"]["pii_masked"] == {"AADHAAR": 1, "PAN": 1}


def test_error_is_recorded_on_the_failing_span():
    class Boom(StreamLLM):
        def complete(self, **kw):
            raise RuntimeError("down")
    tr = Trace("r5", "ask")
    with pytest.raises(RuntimeError):
        RAG(retriever=FakeRetriever(0.85), llm=Boom(), gate=0.67).answer("q?", trace=tr)
    assert next(s for s in tr.to_dict()["spans"] if s["name"] == "llm")["error"] == "RuntimeError"


def test_store_ring_file_rotation_and_summary(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl", ring=3, max_bytes=400)
    rag = make_rag()
    for i in range(5):
        tr = Trace(f"id{i}", "ask")
        store.add(tr.finish(rag.answer("cooling-off?", trace=tr)))
    assert store.get("id0") is None and store.get("id4")["request_id"] == "id4"
    assert [t["request_id"] for t in store.recent(2)] == ["id4", "id3"]
    assert (tmp_path / "t.jsonl.1").exists()
    s = store.summary()
    assert s["requests"] == 3 and s["reasons"] == {"answered": 3} and s["tokens"]["input"] == 3000


def api(tmp_path, admin="adm"):
    app = create_app(rag=make_rag(), api_keys="k1", admin_keys=admin,
                     trace_store=TraceStore(tmp_path / "t.jsonl"), rate_limit_per_min=100)
    return TestClient(app)


def test_admin_endpoints_roundtrip(tmp_path):
    with api(tmp_path) as c:
        r = c.post("/ask", json={"question": "cooling-off period?"}, headers={"X-API-Key": "k1"})
        rid = r.headers["X-Request-ID"]
        t = c.get(f"/traces/{rid}", headers={"X-API-Key": "adm"})
        assert t.status_code == 200 and t.json()["outcome"]["reason"] == "answered"
        assert "retrieval" in names(t.json()) and t.json()["key"] != "k1"  # key is hashed
        assert c.get("/traces", headers={"X-API-Key": "adm"}).json()[0]["request_id"] == rid
        assert c.get("/traces/summary", headers={"X-API-Key": "adm"}).json()["requests"] == 1
        assert c.get("/traces/nope", headers={"X-API-Key": "adm"}).status_code == 404


def test_streamed_request_is_traced(tmp_path):
    with api(tmp_path) as c:
        r = c.post("/ask/stream", json={"question": "cooling-off period?"}, headers={"X-API-Key": "k1"})
        t = c.get(f"/traces/{r.headers['X-Request-ID']}", headers={"X-API-Key": "adm"}).json()
        assert t["endpoint"] == "ask_stream" and t["outcome"]["reason"] == "answered"


def test_regular_api_key_cannot_read_traces(tmp_path):
    with api(tmp_path) as c:
        assert c.get("/traces", headers={"X-API-Key": "k1"}).status_code == 401
        assert c.get("/traces").status_code == 401


def test_traces_disabled_when_no_admin_keys(tmp_path):
    with api(tmp_path, admin="") as c:
        assert c.get("/traces", headers={"X-API-Key": "k1"}).status_code == 403


def test_cached_call_costs_nothing_in_outcome_but_keeps_token_counts():
    class Cached(StreamLLM):
        def _res(self):
            r = super()._res()
            r.cached = True
            return r
    tr = Trace("r6", "ask")
    a = RAG(retriever=FakeRetriever(0.85), llm=Cached(), gate=0.67).answer("cooling-off?", trace=tr)
    o = tr.finish(a).to_dict()["outcome"]
    assert o["cached"] is True and o["cost_usd"] == 0.0 and o["input_tokens"] == 1000
