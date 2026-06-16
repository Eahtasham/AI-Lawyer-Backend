"""
Offline Information-Retrieval benchmark (no LLM generation, no Gemini quota beyond
~one query embedding each).

Compares five retrieval strategies on a labelled query set using TREC-style **pooled
relevance**: candidates from all methods are pooled and judged with a reference matcher,
then standard IR metrics are computed per method.

    dense              Qdrant vector search only (current production)
    bm25               lexical BM25 only
    hybrid_rrf         dense + BM25 fused with Reciprocal Rank Fusion
    hybrid_mmr         + Maximal Marginal Relevance reranking
    hybrid_mmr_auth    + legal-authority PageRank blend (cases)

Metrics (@5 and @10): Precision, Recall, Hit-rate, MRR, nDCG, MAP.

Usage (from backend-ai-lawyer/):
    venv/Scripts/python.exe -m scripts.eval_retrieval
    venv/Scripts/python.exe -m scripts.eval_retrieval --authority-weight 0.25
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.qdrant import qdrant_service  # noqa: E402
from app.services import bm25_index  # noqa: E402
from app.services.hybrid import _mmr_order  # noqa: E402

COLLECTION = {"statutes": settings.QDRANT_COLLECTION_STATUTES,
              "cases": settings.QDRANT_COLLECTION_CASES}
METHODS = ["dense", "bm25", "hybrid_rrf", "hybrid_mmr", "hybrid_rrf_auth"]
KS = [5, 10]
POOL_N = 30  # candidates per method used for pooling + ranking depth


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def is_relevant(text: str, meta: dict, refs: list) -> bool:
    md = meta or {}
    blob = _norm(" ".join([
        text or "", str(md.get("law", "")), str(md.get("section_number", "")),
        str(md.get("section_title", "")), str(md.get("case_name", "")),
        str(md.get("citation", "")), str(md.get("chapter_title", "")),
    ]))
    return any(_norm(r) in blob for r in refs)


# --- metrics ----------------------------------------------------------------
def precision_at_k(rels, k):
    topk = rels[:k]
    return sum(topk) / k if k else 0.0


def recall_at_k(rels, k, total_rel):
    return sum(rels[:k]) / total_rel if total_rel else 0.0


def hit_at_k(rels, k):
    return 1.0 if any(rels[:k]) else 0.0


def reciprocal_rank(rels):
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels, k):
    dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(rels[:k], start=1))
    n_rel = int(round(sum(rels)))
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, n_rel) + 1))
    return dcg / ideal if ideal else 0.0


def average_precision(rels, total_rel):
    if not total_rel:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            score += hits / i
    return score / total_rel


def build_candidates(query, source_type, q_emb, store):
    """Return registry {id: {text, meta, vector}} and per-method rank maps."""
    collection = COLLECTION[source_type]
    dense = qdrant_service.query_with_embedding(q_emb, collection, POOL_N, with_vectors=True)
    sparse = store.top_n(query, POOL_N)

    registry, dense_rank, bm25_rank = {}, {}, {}
    for r, d in enumerate(dense, start=1):
        dense_rank[d["id"]] = r
        registry[d["id"]] = {"text": d["text"], "meta": d["metadata"],
                             "vector": np.asarray(d["vector"], dtype=np.float32) if d["vector"] is not None else None}
    for (cid, score, rank, text, meta) in sparse:
        bm25_rank[cid] = rank
        if cid not in registry:
            registry[cid] = {"text": text, "meta": meta, "vector": None}

    # fill missing vectors for fusion/MMR
    missing = [c for c in registry if registry[c]["vector"] is None]
    if missing:
        for cid, vec in qdrant_service.fetch_vectors(missing, collection).items():
            if vec is not None:
                registry[cid]["vector"] = np.asarray(vec, dtype=np.float32)
    return registry, dense_rank, bm25_rank


def ranked_lists(query, source_type, q_emb, registry, dense_rank, bm25_rank, auth_weight):
    """Produce an ordered id list per method."""
    k_rrf = settings.RRF_K
    rrf = {}
    for cid in registry:
        s = 0.0
        if cid in dense_rank:
            s += 1.0 / (k_rrf + dense_rank[cid])
        if cid in bm25_rank:
            s += 1.0 / (k_rrf + bm25_rank[cid])
        rrf[cid] = s

    lists = {
        "dense": sorted(dense_rank, key=lambda c: dense_rank[c]),
        "bm25": sorted(bm25_rank, key=lambda c: bm25_rank[c]),
        "hybrid_rrf": sorted(registry, key=lambda c: rrf[c], reverse=True),
    }

    # MMR over the RRF pool (diversity-aware reranking; benchmarked as a component)
    q_vec = np.asarray(q_emb, dtype=np.float32)
    pool = [c for c in lists["hybrid_rrf"][: settings.MMR_POOL] if registry[c]["vector"] is not None]
    vec_map = {c: registry[c]["vector"] for c in pool}
    mmr_order, rel = _mmr_order(q_vec, pool, vec_map, settings.MMR_LAMBDA, len(pool))
    lists["hybrid_mmr"] = mmr_order

    # Authority blend on the RRF ranking (the production hybrid order), cases only
    if source_type == "cases" and auth_weight > 0:
        from app.services.authority_graph import authority_service
        rrf_pool = lists["hybrid_rrf"][: settings.MMR_POOL]
        relevance = {c: rel.get(c, _cosine_safe(q_vec, registry, c)) for c in rrf_pool}
        auth = {c: authority_service.score(c) for c in rrf_pool}
        lists["hybrid_rrf_auth"] = sorted(
            rrf_pool, key=lambda c: (1 - auth_weight) * relevance[c] + auth_weight * auth.get(c, 0.0),
            reverse=True) + lists["hybrid_rrf"][settings.MMR_POOL:]
    else:
        lists["hybrid_rrf_auth"] = lists["hybrid_rrf"]
    return lists


def _cosine_safe(q_vec, registry, cid):
    from app.services.hybrid import _cosine
    v = registry[cid]["vector"]
    return _cosine(q_vec, v) if v is not None else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-weight", type=float, default=settings.AUTHORITY_WEIGHT)
    ap.add_argument("--dataset", default=str(ROOT / "eval_data" / "retrieval_eval.json"))
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        queries = json.load(f)

    agg = {m: {f"{name}@{k}": [] for k in KS for name in
               ["precision", "recall", "hit", "ndcg"]} for m in METHODS}
    for m in METHODS:
        agg[m]["mrr"] = []
        agg[m]["map"] = []

    n_eval = 0
    print(f"Evaluating {len(queries)} queries across {len(METHODS)} methods "
          f"(authority_weight={args.authority_weight})...\n")

    for qi, item in enumerate(queries, start=1):
        query, st, refs = item["query"], item["source_type"], item["relevant_refs"]
        store = bm25_index.get_store(st)
        if store is None:
            print("  ! BM25 index missing; aborting."); return
        q_emb = qdrant_service._get_embedding(query)
        registry, dense_rank, bm25_rank = build_candidates(query, st, q_emb, store)

        # pooled relevance judgments
        relset = {cid for cid in registry if is_relevant(registry[cid]["text"], registry[cid]["meta"], refs)}
        if not relset:
            print(f"[{qi}] (skipped, no relevant docs pooled) {query[:50]}")
            continue
        n_eval += 1
        total_rel = len(relset)

        lists = ranked_lists(query, st, q_emb, registry, dense_rank, bm25_rank, args.authority_weight)
        line = f"[{qi}] {query[:46]:46s} |rel|={total_rel} "
        for m in METHODS:
            rels = [1.0 if cid in relset else 0.0 for cid in lists[m]]
            for k in KS:
                agg[m][f"precision@{k}"].append(precision_at_k(rels, k))
                agg[m][f"recall@{k}"].append(recall_at_k(rels, k, total_rel))
                agg[m][f"hit@{k}"].append(hit_at_k(rels, k))
                agg[m][f"ndcg@{k}"].append(ndcg_at_k(rels, k))
            agg[m]["mrr"].append(reciprocal_rank(rels))
            agg[m]["map"].append(average_precision(rels, total_rel))
            line += f"| {m[:6]} nDCG@10={agg[m]['ndcg@10'][-1]:.2f} "
        print(line)

    # aggregate
    summary = {m: {k: (sum(v) / len(v) if v else 0.0) for k, v in agg[m].items()} for m in METHODS}
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    _write_outputs(out_dir, summary, n_eval, args.authority_weight)
    print(f"\nEvaluated {n_eval}/{len(queries)} queries. Reports in {out_dir}/")


def _write_outputs(out_dir, summary, n_eval, auth_w):
    cols = ["precision@5", "recall@5", "ndcg@5", "hit@5",
            "precision@10", "recall@10", "ndcg@10", "hit@10", "mrr", "map"]
    labels = {"dense": "Dense (vector only)", "bm25": "BM25 (lexical only)",
              "hybrid_rrf": "Hybrid (RRF)", "hybrid_mmr": "Hybrid + MMR",
              "hybrid_rrf_auth": "Hybrid (RRF) + Authority"}

    # CSV
    with open(out_dir / "retrieval_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols)
        for m in METHODS:
            w.writerow([m] + [round(summary[m][c], 4) for c in cols])

    # Markdown
    L = ["# Retrieval Benchmark — Hybrid IR Engine", "",
         f"*Pooled-relevance (TREC-style) evaluation over {n_eval} labelled legal queries. "
         f"Authority blend weight = {auth_w}.*", "",
         "Metrics are means across queries. Higher is better for all.", "",
         "| Method | P@5 | Recall@5 | nDCG@5 | Hit@5 | nDCG@10 | MRR | MAP |",
         "|--------|:---:|:--------:|:------:|:-----:|:-------:|:---:|:---:|"]
    for m in METHODS:
        s = summary[m]
        L.append(f"| **{labels[m]}** | {s['precision@5']:.3f} | {s['recall@5']:.3f} | "
                 f"{s['ndcg@5']:.3f} | {s['hit@5']:.3f} | {s['ndcg@10']:.3f} | "
                 f"{s['mrr']:.3f} | {s['map']:.3f} |")
    L += ["", "## Notes",
          "- **Pooled relevance:** candidates from all methods are pooled and judged "
          "relevant if their text/metadata match the query's reference labels — the standard "
          "way to evaluate retrieval without exhaustive human judgments.",
          "- **Dense** = current production (Qdrant cosine). **Hybrid** adds an in-house BM25 "
          "index fused via Reciprocal Rank Fusion, then MMR diversity reranking.",
          "- nDCG/MRR/MAP use binary relevance; P@k and Recall@k are pool-relative.", ""]
    (out_dir / "retrieval_metrics.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
