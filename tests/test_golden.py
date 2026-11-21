import json

import pytest

from rbirag.config import settings

pytestmark = pytest.mark.skipif(not (settings.delta_dir / "silver_chunks").exists(),
                                reason="silver table not built")


def test_golden_labels_match_corpus():
    import sys
    sys.path.insert(0, str(settings.delta_dir.parents[1] / "evals"))
    from validate_golden import GOLDEN, load_chunks, validate

    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    assert len(rows) >= 40
    assert validate(rows, load_chunks()) == []
