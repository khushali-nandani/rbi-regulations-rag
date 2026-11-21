"""Section-aware chunking for RBI directions.

Strategy: split on top-level numbered paragraphs ("9. Loan disbursal ...") and chapter
headings, track page_start/page_end, then split oversize sections at sub-item boundaries
(i. / ii. / a. / (a)) so no chunk loses its parent heading (prefixed to every chunk).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parse import Page, sha

MAX_WORDS = 280
MIN_WORDS = 25

CHAPTER = re.compile(r"^(Chapter\s+[IVXLC]+|CHAPTER\s+[IVXLC]+)\b[:\s-]*(.*)$")
# top-level "12. Heading text" : number, dot, capitalised words, short line, not a sentence end
SECTION = re.compile(r"^(\d{1,3})\.\s*(\S.*)?$")
# Real annex headings: "Annex I - Title", "Annex-II", "Annex". Wrapped cross-references such as
# "Annex IX, sworn before..." or "Annex XIII)" must NOT match (they reset section numbering).
ANNEX = re.compile(r"^(ANNEX(URE)?|Annex(ure)?)(?:[\s-]*[IVXLC\d]+)?(?:\s*[-\u2013\u2014:]\s*\S[^\n]{0,90})?\s*$")
APPENDIX = re.compile(r"^Appendix\s*$")  # repealed-circular tables: not regulatory content
SUBITEM = re.compile(r"^(\(?[ivxl]{1,5}[\.\)]|\(?[a-z][\.\)]|\(\d{1,2}\))\s+\S")
NOISE = re.compile(r"^\s*(\d{1,3}|Page \d+ of \d+)\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chapter: str
    section_no: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    content_hash: str


def _title(text: str) -> str:
    """Short title: text up to first colon/period, capped at 12 words."""
    head = re.split(r"[:.](?:\s|$)", text.strip(), maxsplit=1)[0]
    return " ".join(head.split()[:12]).rstrip(",;")


def _sections(pages: list[Page]):
    """Yield (chapter, no, title, [(page, line), ...]) for each section."""
    chapter, no, title, buf = "", "0", "Front matter", []
    last_no = 0
    for pg in pages:
        if pg.is_toc:
            continue
        for raw in pg.text.splitlines():
            line = raw.strip()
            if not line or NOISE.match(line):
                continue
            if APPENDIX.match(line):
                if buf:
                    yield chapter, no, title, buf
                return
            cm = CHAPTER.match(line)
            if cm:
                chapter = line
                continue
            if ANNEX.match(line):
                if buf:
                    yield chapter, no, title, buf
                chapter, no, title, buf, last_no = line, f"A:{line[:24]}", line, [], 0
                continue
            hm = SECTION.match(line)
            # section numbers must increase monotonically: rejects list items inside paragraphs
            if hm and last_no + 1 <= int(hm.group(1)) <= last_no + 3:
                if buf:
                    yield chapter, no, title, buf
                rest = hm.group(2) or ""
                no, last_no = hm.group(1), int(hm.group(1))
                title, buf = _title(rest) or "Section " + no, ([(pg.page, rest)] if rest else [])
                continue
            buf.append((pg.page, line))
    if buf:
        yield chapter, no, title, buf


def _split(lines: list[tuple[int, str]]):
    """Split a section's lines into groups <= MAX_WORDS, preferring sub-item boundaries."""
    groups, cur, words = [], [], 0
    for page, line in lines:
        w = len(line.split())
        if cur and words + w > MAX_WORDS and SUBITEM.match(line):
            groups.append(cur)
            cur, words = [], 0
        elif cur and words + w > MAX_WORDS * 1.5:  # no sub-item boundary found: hard split
            groups.append(cur)
            cur, words = [], 0
        cur.append((page, line))
        words += w
    if cur:
        groups.append(cur)
    return groups


def chunk_document(doc_id: str, pages: list[Page]) -> list[Chunk]:
    out: list[Chunk] = []
    for chapter, no, title, lines in _sections(pages):
        for group in _split(lines):
            body = " ".join(l for _, l in group)
            if len(body.split()) < MIN_WORDS and out and out[-1].section_no == no:
                prev = out[-1]  # merge tiny tails into previous chunk of the same section
                prev.text += " " + body
                prev.page_end = group[-1][0]
                prev.content_hash = sha(prev.text)
                prev.chunk_id = sha(f"{doc_id}|{no}|{prev.text}")[:16]
                continue
            text = f"[{title}] {body}"
            out.append(Chunk(
                chunk_id=sha(f"{doc_id}|{no}|{text}")[:16], doc_id=doc_id, chapter=chapter,
                section_no=no, section_title=title, page_start=group[0][0],
                page_end=group[-1][0], text=text, content_hash=sha(text),
            ))
    return out
