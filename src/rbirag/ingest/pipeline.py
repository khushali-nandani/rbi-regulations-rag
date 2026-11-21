"""Ingestion: PDFs -> bronze (pages) -> silver (chunks), Delta tables, idempotent via content hash.

Re-running is safe: bronze merges on (doc_id, page) and silver on chunk_id (a hash of the
chunk's content), so unchanged content is never duplicated and edited content replaces
only the chunks whose hash changed.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yaml
from deltalake import DeltaTable, write_deltalake

from .chunk import chunk_document
from .parse import parse_pdf

ROOT = Path(__file__).resolve().parents[3]


def _upsert(path: Path, df: pd.DataFrame, keys: list[str]) -> dict:
    if not (path / "_delta_log").exists():
        write_deltalake(path, df)
        return {"inserted": len(df), "updated": 0}
    pred = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    res = (
        DeltaTable(path)
        .merge(df, predicate=pred, source_alias="s", target_alias="t")
        .when_matched_update_all(predicate="t.content_hash != s.content_hash")
        .when_not_matched_insert_all()
        .execute()
    )
    return {"inserted": res["num_target_rows_inserted"], "updated": res["num_target_rows_updated"]}


def run(manifest: Path = ROOT / "data/manifest.yaml", out: Path = ROOT / "data/delta") -> dict:
    docs = yaml.safe_load(manifest.read_text())["documents"]
    now = dt.datetime.now(dt.UTC).isoformat()
    bronze_rows, silver_rows = [], []
    for d in docs:
        pdf = ROOT / "data/raw" / f"{d['id']}.pdf"
        if not pdf.exists():
            print(f"skip {d['id']}: {pdf.name} missing")
            continue
        pages = parse_pdf(d["id"], str(pdf))
        for p in pages:
            bronze_rows.append({"doc_id": p.doc_id, "page": p.page, "text": p.text,
                                "content_hash": p.content_hash, "is_toc": p.is_toc,
                                "ingested_at": now})
        for c in chunk_document(d["id"], pages):
            silver_rows.append({
                **{k: getattr(c, k) for k in ("chunk_id", "doc_id", "chapter", "section_no",
                   "section_title", "page_start", "page_end", "text", "content_hash")},
                "doc_title": d["title"], "status": d["status"],
                "effective_date": str(d["effective_date"]), "ingested_at": now,
            })
    out.mkdir(parents=True, exist_ok=True)
    stats = {
        "bronze_pages": _upsert(out / "bronze_pages", pd.DataFrame(bronze_rows),
                                ["doc_id", "page"]),
        "silver_chunks": _upsert(out / "silver_chunks", pd.DataFrame(silver_rows), ["chunk_id"]),
    }
    stats["silver_total_rows"] = len(DeltaTable(out / "silver_chunks").to_pandas())
    return stats


if __name__ == "__main__":
    print(run())
