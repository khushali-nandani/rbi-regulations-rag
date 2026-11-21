"""Validate evals/golden/questions.jsonl against the silver chunk table.

For each gold entry, the quote must appear verbatim (whitespace-normalised) in a chunk of that
(doc_id, section_no). Pages are filled in from the matched chunk so labels never drift.
Run with --write to persist the filled-in page ranges.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from deltalake import DeltaTable

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals/golden/questions.jsonl"
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_chunks():
    df = DeltaTable(ROOT / "data/delta/silver_chunks").to_pandas()
    df["norm"] = df.text.map(norm)
    return df


def validate(rows, df) -> list[str]:
    errs, ids = [], set()
    for q in rows:
        if q["id"] in ids:
            errs.append(f"{q['id']}: duplicate id")
        ids.add(q["id"])
        if q["answerable"] and not q["gold"]:
            errs.append(f"{q['id']}: answerable question without gold")
        for g in q["gold"]:
            sub = df[(df.doc_id == g["doc_id"]) & (df.section_no == g["section_no"])]
            hit = sub[sub.norm.str.contains(re.escape(norm(g["quote"])), regex=True)]
            if hit.empty:
                errs.append(f"{q['id']}: quote not found in {g['doc_id']} §{g['section_no']}: "
                            f"{g['quote']!r}")
            else:
                g["pages"] = [int(hit.page_start.min()), int(hit.page_end.max())]
    return errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    errs = validate(rows, load_chunks())
    from collections import Counter
    print(f"{len(rows)} questions:", dict(Counter(r["type"] for r in rows)))
    for e in errs:
        print("FAIL", e)
    if a.write and not errs:
        GOLDEN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print("wrote pages into", GOLDEN.name)
    sys.exit(1 if errs else 0)
