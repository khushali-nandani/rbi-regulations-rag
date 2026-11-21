"""PDF -> per-page text (bronze). Keeps page numbers so citations can point at pages."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pymupdf

_DOT_LEADER = re.compile(r"\.{6,}")


@dataclass(frozen=True)
class Page:
    doc_id: str
    page: int  # 1-indexed, matches the PDF viewer
    text: str
    content_hash: str
    is_toc: bool


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_pdf(doc_id: str, path: str) -> list[Page]:
    pages = []
    with pymupdf.open(path) as d:
        for i, pg in enumerate(d, start=1):
            text = pg.get_text().replace(" ", " ")
            # Index pages are full of dot leaders; they pollute retrieval (match every heading).
            is_toc = len(_DOT_LEADER.findall(text)) >= 4
            pages.append(Page(doc_id, i, text, sha(text), is_toc))
    return pages
