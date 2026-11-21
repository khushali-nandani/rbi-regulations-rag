import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from check_regression import compare  # noqa: E402


def frame(**vals):
    return pd.DataFrame([{"config": "hybrid_rrf", "hit@1": 0.6, "hit@5": 0.9, "mrr@10": 0.77, **vals}])


def test_small_noise_within_tolerance_passes():
    rows, ok = compare(frame(), frame(**{"hit@1": 0.575}), tol=0.03)
    assert ok and {r["status"] for r in rows} <= {"ok", "improved"}


def test_real_drop_fails_and_names_the_metric():
    rows, ok = compare(frame(), frame(**{"mrr@10": 0.70}), tol=0.03)
    assert not ok and [r["metric"] for r in rows if r["status"] == "FAIL"] == ["mrr@10"]


def test_improvement_is_reported_not_failed():
    rows, ok = compare(frame(), frame(**{"hit@5": 0.95}), tol=0.03)
    assert ok and any(r["status"] == "improved" for r in rows)
