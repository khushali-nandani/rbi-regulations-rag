import pytest

from rbirag.ingest import pipeline
from rbirag.ingest.chunk import chunk_document
from rbirag.ingest.parse import parse_pdf

DL = pipeline.ROOT / "data/raw/digital_lending_2025.pdf"
pytestmark = pytest.mark.skipif(not DL.exists(), reason="corpus PDFs not present")


def test_chunks_carry_page_and_section_metadata():
    chunks = chunk_document("digital_lending_2025", parse_pdf("digital_lending_2025", str(DL)))
    assert chunks and all(1 <= c.page_start <= c.page_end <= 24 for c in chunks)
    disb = [c for c in chunks if "Loan disbursal" in c.section_title]
    assert disb and all(c.section_no == "9" for c in disb)


def test_reingest_is_idempotent(tmp_path):
    first = pipeline.run(out=tmp_path)
    second = pipeline.run(out=tmp_path)
    assert first["silver_chunks"]["inserted"] > 0
    assert second["silver_chunks"] == {"inserted": 0, "updated": 0}
    assert second["silver_total_rows"] == first["silver_total_rows"]
