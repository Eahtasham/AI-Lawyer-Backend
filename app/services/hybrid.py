"""
Hybrid retrieval: dense (Qdrant) + BM25 (lexical), fused with Reciprocal Rank Fusion,
reranked with Maximal Marginal Relevance, optionally blended with the legal-authority
PageRank score (cases only).

Returns the standard chunk shape used everywhere else:
    {"rank": int, "score": float in [0,1], "text": str, "metadata": dict}
with an additive metadata["retrieval"] = {dense_rank, bm25_rank, rrf_score, authority}
for transparency / the demo UI.

If the BM25 index is missing, this falls back to plain dense search so the system keeps
working.
"""
import time
from typing import List, Dict, Optional

import numpy as np

from app.config import settings
from app.logger import logger
from app.services.qdrant import qdrant_service
from app.services import bm25_index

_COLLECTION = {
    "statutes": settings.QDRANT_COLLECTION_STATUTES,
    "cases": settings.QDRANT_COLLECTION_CASES,
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mmr_order(q_vec: np.ndarray, cand_ids: List, vec_map: Dict, lambda_: float,
               k: int) -> List:
    """Greedy Maximal Marginal Relevance. Returns ids ordered by MMR selection."""
    remaining = [c for c in cand_ids if c in vec_map]
    rel = {c: _cosine(q_vec, vec_map[c]) for c in remaining}
    selected: List = []
    while remaining and len(selected) < k:
        best, best_val = None, -1e9
        for c in remaining:
            diversity = max((_cosine(vec_map[c], vec_map[s]) for s in selected), default=0.0)
            val = lambda_ * rel[c] - (1 - lambda_) * diversity
            if val > best_val:
                best_val, best = val, c
        selected.append(best)
        remaining.remove(best)
    return selected, rel


def _dense_fallback(query: str, source_type: str, top_k: int) -> List[Dict]:
    fn = qdrant_service.search_cases if source_type == "cases" else qdrant_service.search_statutes
    return fn(query, top_k)


def hybrid_search(query: str, source_type: str, top_k: int = None) -> List[Dict]:
    top_k = top_k or settings.RAG_TOP_K
    collection = _COLLECTION[source_type]

    store = bm25_index.get_store(source_type)
    if store is None:
        logger.warning(f"[Hybrid] {source_type}: BM25 index unavailable -> DENSE fallback")
        return _dense_fallback(query, source_type, top_k)

    try:
        t0 = time.time()
        logger.info(f"[Hybrid] ========== {source_type.upper()} HYBRID RETRIEVAL ==========")
        logger.info(f"[Hybrid] Query: {query[:80]}")

        # 1. Dense retrieval (embed once, reuse for MMR relevance)
        q_emb = qdrant_service._get_embedding(query)
        q_vec = np.asarray(q_emb, dtype=np.float32)
        dense = qdrant_service.query_with_embedding(q_emb, collection, settings.N_DENSE,
                                                    with_vectors=True)
        top_dense = dense[0]["score"] if dense else 0.0
        logger.info(f"[Hybrid] [1] Dense (vector) search  : {len(dense)} candidates "
                    f"(top cosine {top_dense:.3f}, k={settings.N_DENSE})")

        # 2. Sparse retrieval (BM25)
        sparse = store.top_n(query, settings.N_SPARSE)  # (id, score, rank, text, meta)
        top_bm25 = sparse[0][1] if sparse else 0.0
        logger.info(f"[Hybrid] [2] BM25 (lexical) search   : {len(sparse)} candidates "
                    f"(top BM25 {top_bm25:.3f}, k={settings.N_SPARSE})")

        # 3. Registry of all candidates + their content/vectors
        registry: Dict = {}
        dense_rank: Dict = {}
        for r, d in enumerate(dense, start=1):
            dense_rank[d["id"]] = r
            registry[d["id"]] = {
                "text": d["text"], "metadata": d["metadata"],
                "vector": np.asarray(d["vector"], dtype=np.float32) if d["vector"] is not None else None,
            }
        bm25_rank: Dict = {}
        for (cid, score, rank, text, meta) in sparse:
            bm25_rank[cid] = rank
            if cid not in registry:
                registry[cid] = {"text": text, "metadata": meta, "vector": None}

        # 4. Reciprocal Rank Fusion over the union
        k_rrf = settings.RRF_K
        rrf: Dict = {}
        for cid in registry:
            s = 0.0
            if cid in dense_rank:
                s += 1.0 / (k_rrf + dense_rank[cid])
            if cid in bm25_rank:
                s += 1.0 / (k_rrf + bm25_rank[cid])
            rrf[cid] = s
        overlap = len(set(dense_rank) & set(bm25_rank))
        logger.info(f"[Hybrid] [3] RRF fusion (k={k_rrf})     : {len(registry)} unique docs "
                    f"({overlap} found by BOTH retrievers, get a fusion boost)")

        # 5. Take the top pool by RRF for reranking
        pool = sorted(registry.keys(), key=lambda c: rrf[c], reverse=True)[: settings.MMR_POOL]

        # 6. Ensure vectors for the pool (BM25-only candidates need a vector for MMR)
        missing = [c for c in pool if registry[c]["vector"] is None]
        if missing:
            fetched = qdrant_service.fetch_vectors(missing, collection)
            for cid, vec in fetched.items():
                if vec is not None:
                    registry[cid]["vector"] = np.asarray(vec, dtype=np.float32)
        pool = [c for c in pool if registry[c]["vector"] is not None]

        # 7. Final relevance order: RRF (default) or MMR diversity rerank (opt-in)
        vec_map = {c: registry[c]["vector"] for c in pool}
        rel = {c: _cosine(q_vec, vec_map[c]) for c in pool}
        if settings.USE_MMR:
            ordered, rel = _mmr_order(q_vec, pool, vec_map, settings.MMR_LAMBDA,
                                      max(top_k, settings.MMR_POOL))
            logger.info(f"[Hybrid] [4] Rerank: MMR (lambda={settings.MMR_LAMBDA}) over "
                        f"pool of {len(pool)} (diversity-aware)")
        else:
            ordered = pool  # already sorted by RRF score
            logger.info(f"[Hybrid] [4] Rerank: RRF order over pool of {len(pool)} (MMR off)")

        # 8. Optional legal-authority blend (cases only)
        w = settings.AUTHORITY_WEIGHT
        authority: Dict = {c: 0.0 for c in ordered}
        if w > 0 and source_type == "cases":
            from app.services.authority_graph import authority_service
            authority = {c: authority_service.score(c) for c in ordered}
            before = ordered[0] if ordered else None
            ordered = sorted(
                ordered,
                key=lambda c: (1 - w) * rel.get(c, 0.0) + w * authority.get(c, 0.0),
                reverse=True,
            )
            promoted = ordered[0] if ordered else None
            note = "rank-1 unchanged" if promoted == before else "rank-1 changed by authority"
            logger.info(f"[Hybrid] [5] Authority blend (w={w}) : PageRank applied to "
                        f"{len(ordered)} cases ({note})")
        elif source_type == "cases":
            logger.info(f"[Hybrid] [5] Authority blend       : OFF (AUTHORITY_WEIGHT=0)")

        # 9. Build final chunks (score = cosine relevance to query, in ~[0,1])
        final = ordered[:top_k]
        chunks = []
        for i, cid in enumerate(final, start=1):
            entry = registry[cid]
            score = max(0.0, min(1.0, rel.get(cid, 0.0)))
            entry["metadata"]["retrieval"] = {
                "dense_rank": dense_rank.get(cid),
                "bm25_rank": bm25_rank.get(cid),
                "rrf_score": round(rrf.get(cid, 0.0), 6),
                "authority": round(authority.get(cid, 0.0), 6),
            }
            chunks.append({
                "rank": i, "score": score,
                "text": entry["text"], "metadata": entry["metadata"],
            })

        # Final ranked list with provenance (which retriever found each, scores)
        logger.info(f"[Hybrid] [6] Final top-{len(chunks)} ranked results:")
        for c in chunks:
            md = c["metadata"]
            rt = md.get("retrieval", {})
            d_r = rt.get("dense_rank")
            b_r = rt.get("bm25_rank")
            if d_r is not None and b_r is not None:
                src = f"BOTH (D#{d_r},B#{b_r})"
            elif d_r is not None:
                src = f"dense (D#{d_r})"
            else:
                src = f"bm25 (B#{b_r})"
            name = md.get("law") or md.get("case_name") or "Unknown"
            auth = rt.get("authority", 0.0)
            auth_str = f" auth={auth:.3f}" if auth and auth > 0 else ""
            logger.info(f"[Hybrid]     #{c['rank']} score={c['score']:.3f} rrf={rt.get('rrf_score'):.4f} "
                        f"via {src}{auth_str} | {str(name)[:40]}")
        logger.info(f"[Hybrid] ===== Done in {(time.time() - t0) * 1000:.0f}ms "
                    f"(dense {len(dense)} + bm25 {len(sparse)} -> {len(chunks)}) =====")
        return chunks

    except Exception as e:
        logger.error(f"[Hybrid] {source_type} failed ({e}); falling back to dense.")
        return _dense_fallback(query, source_type, top_k)
