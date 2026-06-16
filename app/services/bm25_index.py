"""
BM25 lexical index — build helpers, tokeniser, and a lazy loader.

The tokeniser and the text-synthesis functions here are shared between the offline
builder (scripts/build_bm25_index.py) and the live query path (app/services/hybrid.py)
so that indexing and querying tokenise text identically.

Persisted index format (pickle), one file per collection:
    {
        "bm25":    rank_bm25.BM25Okapi,     # the scorer (idf, doc freqs, lengths)
        "ids":     [qdrant_point_id, ...],  # parallel to the corpus order
        "texts":   [display_text, ...],     # the text shown to the model / UI
        "metas":   [normalised_metadata, ...],
    }
"""
import os
import re
import pickle
from typing import List, Dict, Optional, Tuple

from app.config import settings
from app.logger import logger

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Light stopword list — keep legal terms; only drop high-frequency English glue.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "been", "by", "with", "as", "at", "that", "this", "it", "its", "from",
    "any", "shall", "such", "which", "who", "whom", "under", "may", "not",
}


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation with light stopword removal."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def synthesize_case_text(payload: dict) -> str:
    """Full lexical text for a Supreme Court case (its `text` field is empty)."""
    summ = payload.get("summary", {}) or {}
    parts = [payload.get("title", "")]
    if isinstance(summ, dict):
        parts += [
            summ.get("executive_summary", ""),
            summ.get("facts", ""),
            summ.get("judgment", ""),
            summ.get("outcome", ""),
            " ".join(map(str, summ.get("legal_principles", []) or [])),
            " ".join(map(str, summ.get("keywords", []) or [])),
        ]
        laws = summ.get("laws_applied", []) or []
        parts.append(" ".join(
            (f"{l.get('name') or ''} {l.get('reference') or ''}".strip()
             if isinstance(l, dict) else str(l or ""))
            for l in laws
        ))
    return "\n".join(p for p in parts if p)


def synthesize_statute_text(payload: dict) -> str:
    """Full lexical text for a statute chunk."""
    parts = [
        payload.get("act_title", payload.get("law", "")),
        " ".join(map(str, payload.get("section_numbers", []) or [])),
        " ".join(map(str, payload.get("section_titles", []) or [])),
        payload.get("chapter_name", payload.get("chapter_title", "")),
        payload.get("text", ""),
    ]
    return "\n".join(str(p) for p in parts if p)


def index_path(source_type: str) -> str:
    """source_type in {'statutes', 'cases'}"""
    return os.path.join(settings.BM25_INDEX_DIR, f"bm25_{source_type}.pkl")


class BM25Store:
    """Wraps a persisted BM25 index for query-time scoring."""

    def __init__(self, bm25, ids: List, texts: List[str], metas: List[Dict]):
        self.bm25 = bm25
        self.ids = ids
        self.texts = texts
        self.metas = metas

    def top_n(self, query: str, n: int) -> List[Tuple]:
        """Return [(qdrant_id, score, rank, text, metadata), ...] for the top-n docs."""
        scores = self.bm25.get_scores(tokenize(query))
        # argsort descending, take n
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        out = []
        for rank, i in enumerate(ranked, start=1):
            out.append((self.ids[i], float(scores[i]), rank, self.texts[i], self.metas[i]))
        return out


# --- Lazy loading -----------------------------------------------------------
_CACHE: Dict[str, Optional[BM25Store]] = {}


def get_store(source_type: str) -> Optional[BM25Store]:
    """Load (and cache) the BM25 index for a collection. Returns None if absent."""
    if source_type in _CACHE:
        return _CACHE[source_type]
    path = index_path(source_type)
    if not os.path.exists(path):
        logger.warning(f"[BM25] Index not found at {path} — hybrid will fall back to dense. "
                       f"Run scripts/build_bm25_index.py to create it.")
        _CACHE[source_type] = None
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = BM25Store(data["bm25"], data["ids"], data["texts"], data["metas"])
        logger.info(f"[BM25] Loaded index '{source_type}' ({len(store.ids)} docs)")
        _CACHE[source_type] = store
        return store
    except Exception as e:
        logger.error(f"[BM25] Failed to load index {path}: {e}")
        _CACHE[source_type] = None
        return None
