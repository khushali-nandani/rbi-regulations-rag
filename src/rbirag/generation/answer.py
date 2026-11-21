"""Retrieve -> gate -> generate -> verify citations. Abstains when evidence is missing."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..config import settings
from ..guardrails import check_input, check_output
from ..guardrails.injection import is_injection
from ..observability.tracing import NULL_TRACE, Trace
from ..retrieval.search import Hit, Retriever
from .llm import LLM, LLMResult
from .prompt import ABSTAIN_TOKEN, SYSTEM, build_user_prompt

_HISTORICAL = re.compile(
    r"\b(2021|2022)\b|\b(earlier|previous(ly)?|older?|repealed|superseded|used to|no longer|"
    r"still (in force|valid|applicable|apply))\b", re.I)
_CITE = re.compile(r"\[(\d+)\]")

BLOCKED_MESSAGE = ("I can only answer questions about RBI regulations, and I can't follow "
                   "instructions embedded in a question.")
TOO_LONG_MESSAGE = "That question is too long. Please shorten it to a sentence or two."
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
    snippet: str = ""  # start of the cited passage, so a reader can verify the claim


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
    pii_types: dict = field(default_factory=dict)  # which PII types were masked (never values)
    injection_detected: bool = False
    dropped_hits: int = 0  # excerpts removed because they contained instructions


@dataclass
class _Ctx:
    question: str
    hits: list[Hit]
    prompt: str
    pii: dict
    inj: bool
    dropped: int
    top_score: float


def wants_history(question: str) -> bool:
    return bool(_HISTORICAL.search(question))


class RAG:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None, k: int = 5,
                 gate: float | None = None, model: str | None = None, mode: str = "hybrid",
                 guardrails: bool = True):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()
        self.k, self.model, self.mode = k, model or settings.gen_model, mode
        self.guardrails = guardrails
        self.gate = settings.gate_min_dense_score if gate is None else gate

    # ---- shared pipeline steps ------------------------------------------------------------
    def _prepare(self, question: str, trace: Trace) -> tuple[Answer | None, _Ctx | None]:
        """Guardrails -> out-of-domain gate -> retrieval. Returns a terminal Answer, or a context."""
        pii, inj = {}, False
        if self.guardrails:
            with trace.span("guardrails_input") as sp:
                g = check_input(question)
                sp.set(pii_types=g.pii_types, injection=g.injection, blocked=g.blocked)
            pii, inj = g.pii_types, g.injection
            if g.blocked:
                msg = {"too_long": TOO_LONG_MESSAGE,
                       "empty": "Please ask a question about RBI regulations."}.get(g.blocked, BLOCKED_MESSAGE)
                return Answer(g.masked, msg, True, f"blocked_{g.blocked}", pii_types=pii,
                              injection_detected=inj), None
            question = g.clean  # from here on, only the masked/stripped question exists
        excl = None if wants_history(question) else ["repealed"]
        # Cheap out-of-domain gate: dense cosine of the best chunk (no LLM call needed).
        with trace.span("gate") as sp:
            top = self.retriever.search(question, mode="dense", k=1, exclude_status=excl)
            top_score = top[0].score if top else 0.0
            sp.set(top_dense_score=round(top_score, 3), threshold=self.gate,
                   passed=top_score >= self.gate)
        if top_score < self.gate:
            return Answer(question, ABSTAIN_MESSAGE, True, "out_of_domain", top_dense_score=top_score,
                          pii_types=pii, injection_detected=inj), None

        with trace.span("retrieval") as sp:
            hits = self.retriever.search(question, mode=self.mode, k=self.k, exclude_status=excl)
            sp.set(mode=self.mode, k=self.k, repealed_excluded=bool(excl), chunks=[
                {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "para": h.section_no,
                 "pages": [h.page_start, h.page_end], "status": h.status, "score": round(h.score, 4)}
                for h in hits])
        dropped = 0
        if self.guardrails:  # indirect injection: an excerpt must never carry instructions
            with trace.span("guardrails_excerpts") as sp:
                kept = [h for h in hits if not is_injection(h.text)]
                dropped, hits = len(hits) - len(kept), kept
                sp.set(dropped=dropped)
            if not hits:
                return Answer(question, ABSTAIN_MESSAGE, True, "no_safe_evidence", pii_types=pii,
                              injection_detected=inj, dropped_hits=dropped), None
        ctx = _Ctx(question, hits, build_user_prompt(question, hits), pii, inj, dropped, top_score)
        return None, ctx

    def _finalize(self, ctx: _Ctx, res: LLMResult, trace: Trace) -> Answer:
        with trace.span("verification") as sp:
            a = self._verify(ctx, res)
            sp.set(reason=a.reason, abstained=a.abstained, cited=[c.n for c in a.citations])
        return a

    def _verify(self, ctx: _Ctx, res: LLMResult) -> Answer:
        """Verify the model output: abstain / refusal / leak checks, then citations."""
        hits = ctx.hits
        base = dict(question=ctx.question, hits=hits, llm=res, top_dense_score=ctx.top_score,
                    user_prompt=ctx.prompt, pii_types=ctx.pii, injection_detected=ctx.inj,
                    dropped_hits=ctx.dropped)
        if res.stop_reason == "refusal":
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, reason="refusal", **base)
        text = res.text.strip()
        if ABSTAIN_TOKEN in text:
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, reason="llm_abstained", **base)

        if self.guardrails:
            text, withheld = check_output(text)
            if withheld:
                return Answer(answer=ABSTAIN_MESSAGE, abstained=True, reason="output_blocked", **base)

        cited = sorted({int(n) for n in _CITE.findall(text) if 1 <= int(n) <= len(hits)})
        if not cited:  # an answer with no valid citation is ungrounded: do not show it
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, reason="no_valid_citation", **base)
        # drop markers that point at excerpts that do not exist
        text = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits) else "", text)
        cites = []
        for n in cited:
            h = hits[n - 1]
            pages = f"p.{h.page_start}" if h.page_start == h.page_end else f"pp.{h.page_start}-{h.page_end}"
            cites.append(Citation(n, h.doc_id, h.doc_title, h.section_no, pages, h.status, h.chunk_id,
                                  h.text[:280]))
        return Answer(answer=text, abstained=False, reason="answered", citations=cites, **base)

    # ---- public API -----------------------------------------------------------------------
    def answer(self, question: str, trace: Trace | None = None) -> Answer:
        trace = trace or NULL_TRACE
        terminal, ctx = self._prepare(question, trace)
        if terminal:
            return terminal
        with trace.span("llm", model=self.model) as sp:
            res = self.llm.complete(system=SYSTEM, user=ctx.prompt, model=self.model)
            sp.set(input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                   cost_usd=round(res.cost_usd, 6), cached=res.cached, stop_reason=res.stop_reason)
        return self._finalize(ctx, res, trace)

    def stream(self, question: str, trace: Trace | None = None):
        """Yield {"event", "data"} dicts: token* then one final Answer.

        Tokens are provisional: citations are only verified once generation ends, so the client
        must replace the streamed text with final.answer (it is the abstain message if verification
        fails).
        """
        trace = trace or NULL_TRACE
        terminal, ctx = self._prepare(question, trace)
        if terminal:
            yield {"event": "final", "data": terminal}
            return
        yield {"event": "meta", "data": {"retrieved": len(ctx.hits)}}
        with trace.span("llm", model=self.model, streamed=True) as sp:
            t0, first, n = time.perf_counter(), None, 0
            s = self.llm.stream(system=SYSTEM, user=ctx.prompt, model=self.model)
            for piece in s:
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                n += 1
                yield {"event": "token", "data": piece}
            res = s.result
            sp.set(first_token_ms=round(first, 1) if first is not None else None, token_events=n,
                   input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                   cost_usd=round(res.cost_usd, 6), cached=res.cached, stop_reason=res.stop_reason)
        yield {"event": "final", "data": self._finalize(ctx, res, trace)}
