"""Build the Qdrant index (dense + BM25 sparse vectors) from the silver Delta table.

Rebuilt from scratch each time: embedding ~1k chunks takes seconds, and a rebuild is
trivially deterministic (point ids are uuid5 of chunk_id).
"""
from __future__ import annotations

import uuid

from deltalake import DeltaTable
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from ..config import settings

PAYLOAD_COLS = ["chunk_id", "doc_id", "doc_title", "status", "effective_date", "chapter",
                "section_no", "section_title", "page_start", "page_end", "text"]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def embed_text(row, header: bool) -> str:
    """Contextual header: tell the encoders which document/para a chunk belongs to."""
    return f"{row.doc_title} | para {row.section_no} | {row.text}" if header else row.text


def build_index(collection: str | None = None, header: bool = False, batch: int = 64) -> int:
    collection = collection or settings.collection
    df = DeltaTable(settings.delta_dir / "silver_chunks").to_pandas()
    df = df.sort_values(["doc_id", "page_start", "chunk_id"]).reset_index(drop=True)
    texts = [embed_text(r, header) for r in df.itertuples()]

    cache = str(settings.model_cache)
    dense = TextEmbedding(settings.dense_model, cache_dir=cache)
    sparse = SparseTextEmbedding(settings.sparse_model, cache_dir=cache)
    dense_vecs = list(dense.embed(texts, batch_size=batch))
    sparse_vecs = list(sparse.embed(texts, batch_size=batch))

    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(settings.qdrant_path))
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(
            size=len(dense_vecs[0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    points = [
        models.PointStruct(
            id=point_id(row.chunk_id),
            vector={
                "dense": dv.tolist(),
                "bm25": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
            },
            payload={c: getattr(row, c) for c in PAYLOAD_COLS},
        )
        for row, dv, sv in zip(df.itertuples(), dense_vecs, sparse_vecs)
    ]
    client.upload_points(collection, points, batch_size=batch)
    client.close()
    return len(points)


if __name__ == "__main__":
    print("indexed", build_index(), "chunks (raw)")
    print("indexed", build_index(settings.collection + "_ctx", header=True), "chunks (ctx header)")
