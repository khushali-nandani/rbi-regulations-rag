"""Retrieval modes used for the ablation: dense | bm25 | hybrid (RRF), each optionally reranked."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from ..config import settings

Mode = Literal["dense", "bm25", "hybrid"]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    status: str
    section_no: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    score: float
    rank: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        pages = (f"p.{self.page_start}" if self.page_start == self.page_end
                 else f"pp.{self.page_start}-{self.page_end}")
        return f"{self.doc_id}, para {self.section_no}, {pages}"


class Retriever:
    def __init__(self, qdrant_path=None):
        cache = str(settings.model_cache)
        self.client = QdrantClient(path=str(qdrant_path or settings.qdrant_path))
        self.dense = TextEmbedding(settings.dense_model, cache_dir=cache)
        self.sparse = SparseTextEmbedding(settings.sparse_model, cache_dir=cache)
        self._reranker = None  # loaded lazily: only needed when rerank=True

    @property
    def reranker(self) -> TextCrossEncoder:
        if self._reranker is None:
            self._reranker = TextCrossEncoder(settings.rerank_model,
                                              cache_dir=str(settings.model_cache))
        return self._reranker

    # ---- filters -------------------------------------------------------------------------
    @staticmethod
    def _filter(doc_ids: list[str] | None, exclude_status: list[str] | None):
        must, must_not = [], []
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
        if exclude_status:  # e.g. ["repealed"]: keep superseded rules out of the answer
            must_not.append(models.FieldCondition(
                key="status", match=models.MatchAny(any=exclude_status)))
        return models.Filter(must=must, must_not=must_not) if (must or must_not) else None

    # ---- first-stage candidates ----------------------------------------------------------
    def _candidates(self, query: str, mode: Mode, limit: int, flt) -> list[models.ScoredPoint]:
        dq = list(self.dense.query_embed(query))[0].tolist()
        sq = list(self.sparse.query_embed(query))[0]
        sparse_q = models.SparseVector(indices=sq.indices.tolist(), values=sq.values.tolist())
        c, col = self.client, settings.collection
        if mode == "dense":
            return c.query_points(col, query=dq, using="dense", limit=limit,
                                  query_filter=flt, with_payload=True).points
        if mode == "bm25":
            return c.query_points(col, query=sparse_q, using="bm25", limit=limit,
                                  query_filter=flt, with_payload=True).points
        # hybrid: each retriever fetches `limit`, Qdrant fuses with Reciprocal Rank Fusion
        return c.query_points(
            col,
            prefetch=[
                models.Prefetch(query=dq, using="dense", limit=limit, filter=flt),
                models.Prefetch(query=sparse_q, using="bm25", limit=limit, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, query_filter=flt, with_payload=True,
        ).points

    # ---- public API ----------------------------------------------------------------------
    def search(self, query: str, *, mode: Mode = "hybrid", k: int = 5, rerank: bool = False,
               candidates: int = 30, doc_ids: list[str] | None = None,
               exclude_status: list[str] | None = None) -> list[Hit]:
        flt = self._filter(doc_ids, exclude_status)
        points = self._candidates(query, mode, candidates if rerank else k, flt)
        hits = [Hit(**{f: p.payload[f] for f in (
            "chunk_id", "doc_id", "doc_title", "status", "section_no", "section_title",
            "page_start", "page_end", "text")}, score=float(p.score)) for p in points]
        if rerank and hits:
            scores = list(self.reranker.rerank(query, [h.text for h in hits]))
            for h, s in zip(hits, scores):
                h.meta["first_stage_score"], h.score = h.score, float(s)
            hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k]
        for i, h in enumerate(hits, 1):
            h.rank = i
        return hits
