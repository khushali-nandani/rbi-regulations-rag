"""CI gate: re-run the retrieval eval (no LLM, no cost) and fail if quality dropped.

Compares hit@1, hit@5 and MRR@10 of the fast configs against the committed baseline
(evals/runs/retrieval_latest.csv). A drop of more than TOLERANCE on any metric fails the build.
One golden question is worth 0.025, so the default tolerance allows a single flipped question
from cross-platform floating point noise but not a real regression.

If the change is intentional (new chunking, new golden labels), regenerate and commit the baseline:
    python -m rbirag.evals_retrieval
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from rbirag.evals_retrieval import CONFIGS, run_all

ROOT = Path(__file__).resolve().parents[1]
CI_CONFIGS = ("naive_dense", "bm25", "hybrid_rrf")  # reranker configs are too slow for CI
METRICS = ("hit@1", "hit@5", "mrr@10")


def compare(baseline: pd.DataFrame, current: pd.DataFrame, tol: float) -> tuple[list[dict], bool]:
    base = baseline.set_index("config")
    rows, ok = [], True
    for _, r in current.iterrows():
        for m in METRICS:
            b, c = float(base.loc[r["config"], m]), float(r[m])
            drop = b - c
            status = "FAIL" if drop > tol else ("improved" if c > b + 1e-9 else "ok")
            ok &= status != "FAIL"
            rows.append({"config": r["config"], "metric": m, "baseline": b, "current": c,
                         "delta": round(c - b, 4), "status": status})
    return rows, ok


def markdown(rows: list[dict], tol: float) -> str:
    out = [f"### Retrieval eval regression check (tolerance {tol})", "",
           "| config | metric | baseline | current | delta | status |", "|---|---|---|---|---|---|"]
    for r in rows:
        mark = {"FAIL": "❌ FAIL", "improved": "✅ improved", "ok": "✅ ok"}[r["status"]]
        out.append(f"| {r['config']} | {r['metric']} | {r['baseline']:.3f} | {r['current']:.3f} "
                   f"| {r['delta']:+.3f} | {mark} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "evals/runs/retrieval_latest.csv"))
    ap.add_argument("--tolerance", type=float, default=0.03)
    a = ap.parse_args()

    baseline = pd.read_csv(a.baseline)
    configs = [c for c in CONFIGS if c.name in CI_CONFIGS]
    current = run_all(configs=configs, track=False)
    rows, ok = compare(baseline, current, a.tolerance)
    md = markdown(rows, a.tolerance)
    print(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(summary).write_text(md, encoding="utf-8")
    if not ok:
        print("Retrieval quality regressed. If this is intentional, regenerate the baseline with "
              "`python -m rbirag.evals_retrieval` and commit evals/runs/retrieval_latest.csv.",
              file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
