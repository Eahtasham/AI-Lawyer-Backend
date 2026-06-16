"""
Retrieval dispatcher — selects the retrieval strategy based on settings.RETRIEVAL_MODE.

    "dense"  -> delegate to qdrant_service (current production behaviour, unchanged)
    "hybrid" -> dense (Qdrant) + BM25 (lexical), fused with Reciprocal Rank Fusion,
                reranked with Maximal Marginal Relevance, optionally blended with the
                legal-authority PageRank score for cases.

Both paths return the SAME chunk shape the rest of the system expects:
    {"rank": int, "score": float in [0,1], "text": str, "metadata": dict}

This module is the single integration seam: council.py calls retrieval_service instead
of qdrant_service. With RETRIEVAL_MODE="dense" the output is byte-for-byte identical to
the previous direct qdrant_service calls.
"""
from typing import List, Dict

from app.config import settings
from app.logger import logger
from app.services.qdrant import qdrant_service


class RetrievalService:
    def __init__(self):
        self.mode = (settings.RETRIEVAL_MODE or "dense").lower()
        logger.info(f"[Retrieval] Mode: {self.mode}")

    # --- Public API (mirrors qdrant_service signatures) ----------------------
    def search_statutes(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or settings.RAG_TOP_K
        if self.mode == "hybrid":
            from app.services.hybrid import hybrid_search
            return hybrid_search(query, "statutes", top_k)
        return qdrant_service.search_statutes(query, top_k)

    def search_cases(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or settings.RAG_TOP_K
        if self.mode == "hybrid":
            from app.services.hybrid import hybrid_search
            return hybrid_search(query, "cases", top_k)
        return qdrant_service.search_cases(query, top_k)


retrieval_service = RetrievalService()
