import pytest

from rbirag.config import settings

pytestmark = pytest.mark.skipif(not settings.qdrant_path.exists(),
                                reason="index not built (run python -m rbirag.retrieval.index)")


@pytest.fixture(scope="module")
def retriever():
    from rbirag.retrieval.search import Retriever
    return Retriever()


def test_hybrid_finds_disbursal_para(retriever):
    hits = retriever.search("loan disbursed into LSP account", mode="hybrid", k=3,
                            exclude_status=["repealed"])
    assert hits[0].doc_id == "digital_lending_2025" and hits[0].section_no == "9"


def test_exclude_repealed_and_doc_filter(retriever):
    q = "compensation the ombudsman can award"
    assert any(h.status == "repealed" for h in retriever.search(q, k=10, mode="hybrid"))
    hits = retriever.search(q, k=10, mode="hybrid", exclude_status=["repealed"])
    assert hits and all(h.status != "repealed" for h in hits)
    only = retriever.search(q, k=5, mode="bm25", doc_ids=["ombudsman_2026"])
    assert only and {h.doc_id for h in only} == {"ombudsman_2026"}
