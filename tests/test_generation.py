from rbirag.generation.answer import RAG, wants_history
from rbirag.generation.llm import LLMResult
from rbirag.retrieval.search import Hit


def hit(i, score=0.8, status="current", doc="dl25"):
    return Hit(chunk_id=f"c{i}", doc_id=doc, doc_title=f"Doc {doc}", status=status,
               section_no=str(i), section_title="t", page_start=i, page_end=i,
               text=f"text {i}", score=score)


class FakeRetriever:
    def __init__(self, dense_score=0.8, n=3):
        self.dense_score, self.n, self.calls = dense_score, n, []

    def search(self, q, mode="hybrid", k=5, exclude_status=None, **kw):
        self.calls.append((mode, exclude_status))
        return [hit(i, self.dense_score if mode == "dense" else 1.0) for i in range(1, self.n + 1)][:k]


class FakeLLM:
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.n = text, stop, 0

    def complete(self, **kw):
        self.n += 1
        return LLMResult(text=self.text, stop_reason=self.stop)


def rag(text, dense=0.8, stop="end_turn"):
    return RAG(retriever=FakeRetriever(dense), llm=FakeLLM(text, stop), gate=0.67)


def test_grounded_answer_returns_page_citations():
    a = rag("Not less than one day [2].").answer("cooling-off period?")
    assert not a.abstained and a.reason == "answered"
    assert [(c.n, c.section_no, c.pages) for c in a.citations] == [(2, "2", "p.2")]


def test_llm_abstention():
    a = rag("INSUFFICIENT_EVIDENCE").answer("what is the repo rate?")
    assert a.abstained and a.reason == "llm_abstained"


def test_out_of_domain_skips_llm_call():
    r = rag("anything", dense=0.5)
    a = r.answer("poem about monsoon")
    assert a.abstained and a.reason == "out_of_domain" and r.llm.n == 0


def test_answer_without_valid_citation_is_withheld():
    assert rag("It is one day.").answer("q").reason == "no_valid_citation"
    a = rag("Invented [9].").answer("q")  # cites an excerpt that does not exist
    assert a.abstained and a.reason == "no_valid_citation"


def test_invalid_marker_stripped_but_valid_kept():
    a = rag("Fact A [1]. Fact B [9].").answer("q")
    assert "[9]" not in a.answer and [c.n for c in a.citations] == [1]


def test_refusal_is_abstention():
    assert rag("", stop="refusal").answer("q").reason == "refusal"


def test_repealed_excluded_unless_question_is_historical():
    assert not wants_history("What is the cooling-off period?")
    assert wants_history("Under the 2021 scheme what was the compensation?")
    r = rag("ok [1]")
    r.answer("cooling-off period?")
    assert all(excl == ["repealed"] for _, excl in r.retriever.calls)
    r2 = rag("ok [1]")
    r2.answer("what did the 2022 circular say?")
    assert all(excl is None for _, excl in r2.retriever.calls)
