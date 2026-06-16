"""
Loads precomputed legal-authority PageRank scores (built by
scripts/build_authority_graph.py) for re-ranking retrieved cases.

Lazy, fail-safe: if the artefact is missing, every lookup returns 0.0 so the
retriever behaves exactly as if authority blending were disabled.
"""
import json
import os
from typing import Dict, Optional

from app.config import settings
from app.logger import logger


class AuthorityService:
    def __init__(self):
        self._scores: Optional[Dict[str, float]] = None

    def _load(self):
        path = settings.AUTHORITY_SCORES_PATH
        if not os.path.exists(path):
            logger.warning(f"[Authority] {path} not found — authority blend disabled "
                           f"(run scripts/build_authority_graph.py to create it).")
            self._scores = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._scores = json.load(f)
            logger.info(f"[Authority] Loaded {len(self._scores):,} case authority scores")
        except Exception as e:
            logger.error(f"[Authority] Failed to load {path}: {e}")
            self._scores = {}

    def score(self, case_id) -> float:
        if self._scores is None:
            self._load()
        return float(self._scores.get(str(case_id), 0.0))


authority_service = AuthorityService()
