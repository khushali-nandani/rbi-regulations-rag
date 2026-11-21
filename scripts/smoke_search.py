"""No-LLM sanity check: compare retrieval modes side by side on a few questions."""
import sys

from rbirag.retrieval.search import Retriever

QUERIES = [
    "Can the loan be disbursed into the LSP's account?",
    "Within how many working days must the bank credit back an unauthorised electronic transaction?",
    "Is video based customer identification process (V-CIP) permitted for KYC?",
    "cooling-off period for a digital loan",
    "What compensation can the ombudsman award to a complainant?",
    "Who is responsible for the actions of a service provider to which an NBFC outsources?",
]
r = Retriever()
only = sys.argv[1:] or QUERIES
for q in only:
    print(f"\n### {q}")
    for name, kw in [("dense", dict(mode="dense")), ("bm25", dict(mode="bm25")),
                     ("hybrid", dict(mode="hybrid")), ("hybrid+rerank", dict(mode="hybrid", rerank=True))]:
        hits = r.search(q, k=3, **kw)
        print(f"  {name:14}", " | ".join(f"{h.doc_id}§{h.section_no} p{h.page_start} ({h.score:.2f})" for h in hits))
