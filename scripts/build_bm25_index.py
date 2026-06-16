"""
Build BM25 lexical indexes for both Qdrant collections (offline, no API cost).

Scrolls every point, synthesises lexical text + a display text + normalised metadata,
tokenises, fits a BM25Okapi scorer, and pickles one index per collection to
settings.BM25_INDEX_DIR (default: data/).

Usage (from backend-ai-lawyer/):
    venv/Scripts/python.exe -m scripts.build_bm25_index
    venv/Scripts/python.exe -m scripts.build_bm25_index --collection cases
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rank_bm25 import BM25Okapi          # noqa: E402
from tqdm import tqdm                     # noqa: E402

from app.config import settings           # noqa: E402
from app.services.qdrant import qdrant_service  # noqa: E402
from app.services.bm25_index import (     # noqa: E402
    tokenize, synthesize_case_text, synthesize_statute_text, index_path,
)

COLLECTIONS = {
    "statutes": settings.QDRANT_COLLECTION_STATUTES,
    "cases": settings.QDRANT_COLLECTION_CASES,
}


def _display_text(payload: dict, source_type: str) -> str:
    """Text shown to the model / UI — matches qdrant_service.search() output."""
    text = payload.get("text", "") or ""
    if not text and source_type == "cases":
        summ = payload.get("summary", {}) or {}
        if isinstance(summ, dict):
            text = (f"HEADNOTE: {summ.get('executive_summary', '')}\n\n"
                    f"FACTS: {summ.get('facts', '')}\n\n"
                    f"HELD: {summ.get('judgment', '')}")
    return text


def build_collection(source_type: str):
    collection = COLLECTIONS[source_type]
    print(f"\n=== Building BM25 index for '{source_type}' ({collection}) ===")
    client = qdrant_service.client

    ids, texts, metas, tokenized = [], [], [], []
    next_off = None
    t0 = time.time()
    pbar = tqdm(desc=f"scroll {source_type}", unit="doc")
    while True:
        points, next_off = client.scroll(
            collection_name=collection, limit=1000, offset=next_off,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            lexical = (synthesize_case_text(payload) if source_type == "cases"
                       else synthesize_statute_text(payload))
            toks = tokenize(lexical)
            if not toks:
                continue
            ids.append(p.id)
            tokenized.append(toks)
            texts.append(_display_text(payload, source_type))
            metas.append(qdrant_service._normalize_metadata(payload))
        pbar.update(len(points))
        if next_off is None:
            break
    pbar.close()

    print(f"  Fitting BM25Okapi over {len(tokenized):,} documents...")
    bm25 = BM25Okapi(tokenized)

    os.makedirs(settings.BM25_INDEX_DIR, exist_ok=True)
    out_path = index_path(source_type)
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "texts": texts, "metas": metas}, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved {out_path} ({size_mb:.1f} MB, {len(ids):,} docs, {time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", choices=["statutes", "cases", "all"], default="all")
    args = ap.parse_args()
    targets = ["statutes", "cases"] if args.collection == "all" else [args.collection]
    for t in targets:
        build_collection(t)
    print("\nDone.")


if __name__ == "__main__":
    main()
