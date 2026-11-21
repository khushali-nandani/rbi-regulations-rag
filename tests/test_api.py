import json

import pytest
from fastapi.testclient import TestClient

from rbirag.api.app import create_app
from rbirag.api.security import ApiKeys, RateLimiter
from rbirag.generation.answer import Answer, Citation
from rbirag.generation.llm import BudgetExceeded, LLMResult


class FakeRAG:
    class retriever:  # noqa: N801 - minimal stand-in for health()
        collection = "x"

        class client:  # noqa: N801
            @staticmethod
            def count(_):
                class R:
                    count = 949
                return R()

    def __init__(self, boom=None):
        self.boom, self.seen = boom, []

    def _ans(self, q):
        if self.boom:
            raise self.boom
        self.seen.append(q)
        cite = Citation(1, "digital_lending_2025", "Digital Lending Directions, 2025", "10", "p.10",
                        "current", "abc", "Cooling-off period ...")
        return Answer(q, "Not less than one day [1].", False, "answered", [cite],
                      llm=LLMResult(text="x", input_tokens=100, output_tokens=20, cost_usd=0.0002,
                                    model="claude-haiku-5-5"), pii_types={"PAN": 1})

    def answer(self, q, trace=None):
        return self._ans(q)

    def stream(self, q, trace=None):
        a = self._ans(q)
        yield {"event": "meta", "data": {"retrieved": 5}}
        yield {"event": "token", "data": "Not less "}
        yield {"event": "token", "data": "than one day [1]."}
        yield {"event": "final", "data": a}


def client(rag=None, limit=100):
    app = create_app(rag=rag or FakeRAG(), api_keys="k1,k2", rate_limit_per_min=limit)
    return TestClient(app)


H = {"X-API-Key": "k1"}


def test_health_is_public_and_reports_index():
    with client() as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["chunks_indexed"] == 949 and r.json()["status"] == "ok"
        assert "X-Request-ID" in r.headers


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_ask_requires_valid_api_key(headers):
    with client() as c:
        r = c.post("/ask", json={"question": "cooling off?"}, headers=headers)
        assert r.status_code == 401


def test_ask_returns_grounded_answer_with_citations():
    with client() as c:
        r = c.post("/ask", json={"question": "cooling-off period?"}, headers={"X-API-Key": "k2"})
        d = r.json()
        assert r.status_code == 200 and d["abstained"] is False
        assert d["citations"][0]["doc_id"] == "digital_lending_2025" and d["citations"][0]["pages"] == "p.10"
        assert d["guardrails"]["pii_masked"] == {"PAN": 1} and d["usage"]["model"] == "claude-haiku-5-5"
        assert d["request_id"] == r.headers["X-Request-ID"]


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "x" * 1001}, {"q": "hi"}])
def test_input_validation(body):
    with client() as c:
        assert c.post("/ask", json=body, headers=H).status_code == 422


def test_rate_limit_per_key_with_retry_after():
    with client(limit=2) as c:
        ok = [c.post("/ask", json={"question": "q?"}, headers=H).status_code for _ in range(2)]
        blocked = c.post("/ask", json={"question": "q?"}, headers=H)
        other_key = c.post("/ask", json={"question": "q?"}, headers={"X-API-Key": "k2"})
        assert ok == [200, 200] and blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1 and other_key.status_code == 200


def test_stream_emits_sse_tokens_then_verified_final():
    with client() as c:
        r = c.post("/ask/stream", json={"question": "cooling-off?"}, headers=H)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [(b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:]))
                  for b in r.text.strip().split("\n\n")]
        assert [e for e, _ in events] == ["meta", "token", "token", "final"]
        assert events[-1][1]["citations"][0]["section_no"] == "10"


def test_stream_requires_auth():
    with client() as c:
        assert c.post("/ask/stream", json={"question": "q?"}).status_code == 401


def test_budget_exhausted_is_503_not_500():
    with client(FakeRAG(boom=BudgetExceeded("limit"))) as c:
        r = c.post("/ask", json={"question": "q?"}, headers=H)
        assert r.status_code == 503 and "budget" in r.json()["detail"].lower()


def test_rate_limiter_window_and_keys_unit():
    t = [0.0]
    rl = RateLimiter(2, window=10, clock=lambda: t[0])
    assert rl.check("a")[0] and rl.check("a")[0] and not rl.check("a")[0]
    t[0] = 10.1
    assert rl.check("a")[0]
    ks = ApiKeys(" a , b ,")
    assert ks.valid("a") and ks.valid("b") and not ks.valid("c") and not ks.valid(None)
