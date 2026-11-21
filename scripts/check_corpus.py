"""Verify data/raw PDFs match data/manifest.yaml and are real PDFs (not bot-challenge HTML)."""
import sys
from pathlib import Path

import pymupdf
import yaml

root = Path(__file__).resolve().parents[1]
manifest = yaml.safe_load((root / "data/manifest.yaml").read_text())
bad = 0
for d in manifest["documents"]:
    p = root / "data/raw" / f"{d['id']}.pdf"
    if not p.exists():
        print(f"MISSING  {p.name}")
        bad += 1
        continue
    if p.read_bytes()[:4] != b"%PDF":
        print(f"NOT-PDF  {p.name} (probably a bot-challenge page)")
        bad += 1
        continue
    n = len(pymupdf.open(p))
    chars = sum(len(pg.get_text()) for pg in pymupdf.open(p))
    print(f"OK       {p.name}: {n} pages, {chars} chars")
sys.exit(1 if bad else 0)
