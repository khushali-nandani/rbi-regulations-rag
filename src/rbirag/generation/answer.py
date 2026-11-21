"""Retrieve -> gate -> generate -> verify citations. Abstains when evidence is missing."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import settings
from ..retrieval.search import Hit, Retriever
from .llm import LLM, LLMResult
from .prompt import ABSTAIN_TOKEN, SYSTEM, build_user_prompt

_HISTORICAL = re.compile(
    r"\b(2021|2022)\b|\b(earlier|previous(ly)?|older?|repealed|superseded|used to|no longer|"
    r"still (in force|valid|applicable|apply))\b", re.I)
_CITE = re.compile(r"\[(\d+)\]")

ABSTAIN_MESSAGE = ("I could not find this in the RBI documents I have, so I can't answer it "
                   "reliably. Try rephrasing, or check the source directions on rbi.org.in.")


@dataclass
class Citation:
    n: int
    doc_id: str
    doc_title: str
    section_no: str
    pages: str
    status: str
    chunk_id: str


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    reason: str  # "answered" | "llm_abstained" | "out_of_domain" | "no_valid_citation" | "refusal"
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    llm: LLMResult | None = None
    top_dense_score: float | None = None
    user_prompt: str = ""  # exactly what was sent to the LLM (for PII / leak checks)


def wants_history(question: str) -> bool:
    return bool(_HISTORICAL.search(question))


class RAG:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None, k: int = 5,
                 gate: float | None = None, model: str | None = None, mode: str = "hybrid"):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()
        self.k, self.model, self.mode = k, model or settings.gen_model, mode
        self.gate = settings.gate_min_dense_score if gate is None else gate

    def answer(self, question: str) -> Answer:
        excl = None if wants_history(question) else ["repealed"]
        # Cheap out-of-domain gate: dense cosine of the best chunk (no LLM call needed).
        top = self.retriever.search(question, mode="dense", k=1, exclude_status=excl)
        top_score = top[0].score if top else 0.0
        if top_score < self.gate:
            return Answer(question, ABSTAIN_MESSAGE, True, "out_of_domain", top_dense_score=top_score)

        hits = self.retriever.search(question, mode=self.mode, k=self.k, exclude_status=excl)
        prompt = build_user_prompt(question, hits)
        res = self.llm.complete(system=SYSTEM, user=prompt, model=self.model)
        base = dict(question=question, hits=hits, llm=res, top_dense_score=top_score,
                    user_prompt=prompt)
        if res.stop_reason == "refusal":
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, reason="refusal", **base)
        text = res.text.strip()
        if ABSTAIN_TOKEN in text:
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, reason="llm_abstained", **base)

        cited = sorted({int(n) for n in _CITE.findall(text) if 1 <= int(n) <= len(hits)})
        if not cited:  # an answer with no valid citation is ungrounded: do not show it
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, reason="no_valid_citation", **base)
        # drop markers that point at excerpts that do not exist
        text = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits) else "", text)
        cites = []
        for n in cited:
            h = hits[n - 1]
            pages = f"p.{h.page_start}" if h.page_start == h.page_end else f"pp.{h.page_start}-{h.page_end}"
            cites.append(Citation(n, h.doc_id, h.doc_title, h.section_no, pages, h.status, h.chunk_id))
        return Answer(answer=text, abstained=False, reason="answered", citations=cites, **base)
