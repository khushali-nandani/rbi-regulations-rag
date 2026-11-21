"""Retrieval evaluation: hit@k and MRR against paragraph-level gold labels, tracked in MLflow.

A retrieved chunk is a hit if it has the same doc_id and section_no as a gold entry AND its page
range overlaps the gold page range. Labels are paragraph-level (not chunk ids) so they survive
re-chunking.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ROOT
from .retrieval.search import Hit, Retriever

GOLDEN = ROOT / "evals/golden/questions.jsonl"
KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Config:
    name: str
    mode: str = "hybrid"
    rerank: bool = False
    version_filter: bool = True  # drop repealed docs unless the question is about them
    candidates: int = 30
    collection: str = "rbi_chunks"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"


CONFIGS = [
    Config("naive_dense", mode="dense"),
    Config("bm25", mode="bm25"),
    Config("hybrid_rrf", mode="hybrid"),
    Config("hybrid_rrf_rerank", mode="hybrid", rerank=True),
    Config("hybrid_rrf_rerank_no_version_filter", mode="hybrid", rerank=True, version_filter=False),
    Config("ctx_dense", mode="dense", collection="rbi_chunks_ctx"),
    Config("ctx_hybrid_rrf", mode="hybrid", collection="rbi_chunks_ctx"),
    Config("ctx_hybrid_rrf_rerank", mode="hybrid", rerank=True, collection="rbi_chunks_ctx"),
]


def load_questions(types_excluded=("guardrail",)) -> list[dict]:
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["answerable"] and r["gold"] and r["type"] not in types_excluded]


def is_hit(h: Hit, gold: list[dict]) -> bool:
    for g in gold:
        lo, hi = g["pages"]
        if h.doc_id == g["doc_id"] and h.section_no == g["section_no"] \
                and h.page_start <= hi and h.page_end >= lo:
            return True
    return False


def evaluate(cfg: Config, retriever: Retriever, questions: list[dict], k: int = 10):
    rows = []
    for q in questions:
        retriever.collection = cfg.collection
        excl = ["repealed"] if (cfg.version_filter and not q["include_repealed"]) else None
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], mode=cfg.mode, k=k, rerank=cfg.rerank,
                                candidates=cfg.candidates, rerank_model=cfg.rerank_model,
                                exclude_status=excl)
        ms = (time.perf_counter() - t0) * 1000
        flags = [is_hit(h, q["gold"]) for h in hits]
        rank = next((i for i, f in enumerate(flags, 1) if f), None)
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "first_hit_rank": rank,
            "mrr": 1 / rank if rank else 0.0, "latency_ms": ms,
            "repealed_in_top3": any(h.status == "repealed" for h in hits[:3])
                                 and not q["include_repealed"],
            "top3": " | ".join(f"{h.doc_id}§{h.section_no}" for h in hits[:3]),
            **{f"hit@{kk}": bool(rank and rank <= kk) for kk in KS},
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    m = {f"hit@{k}": df[f"hit@{k}"].mean() for k in KS}
    m["mrr@10"] = df["mrr"].mean()
    m["repealed_in_top3_rate"] = df["repealed_in_top3"].mean()
    m["latency_ms_p50"] = df["latency_ms"].median()
    m["latency_ms_p95"] = df["latency_ms"].quantile(0.95)
    return {k: float(v) for k, v in m.items()}


def run_all(configs=CONFIGS, experiment="retrieval-ablation", track=True) -> pd.DataFrame:
    import mlflow

    questions = load_questions()
    retriever = Retriever()
    retriever.search("warm up", rerank=any(c.rerank for c in configs))  # load models before timing
    if track:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment(experiment)
    table = []
    for cfg in configs:
        df = evaluate(cfg, retriever, questions)
        m = summarise(df)
        if track:
            with mlflow.start_run(run_name=cfg.name):
                mlflow.log_params({**cfg.__dict__, "n_questions": len(questions),
                                   "dense_model": "bge-small-en-v1.5", "chunking": "section_v1"})
                mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in m.items()})
                out = Path(ROOT / f".cache/{cfg.name}_per_question.csv")
                out.parent.mkdir(exist_ok=True)
                df.to_csv(out, index=False)
                mlflow.log_artifact(str(out))
        table.append({"config": cfg.name, **m})
        print(f"done {cfg.name}", flush=True)
    return pd.DataFrame(table)


if __name__ == "__main__":
    res = run_all()
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    out = ROOT / "evals/runs/retrieval_latest.csv"  # committed baseline for the CI regression check
    out.parent.mkdir(exist_ok=True)
    res.round(4).to_csv(out, index=False)
