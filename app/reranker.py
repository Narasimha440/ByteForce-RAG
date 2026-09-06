"""
Local Reranker Modules for SIH 26117.

Supports:
- Cross-Encoder Reranker using sentence-transformers (e.g. BAAI/bge-reranker-v2-m3, MiniLM)
- Fast Lexical Density Reranker for ultra-lightweight CPU execution
- Mock Reranker for unit testing
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for candidate rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank candidates based on joint relevance to query."""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Local Cross-Encoder Reranker using SentenceTransformers.
    Performs joint query-document cross-attention for maximum precision.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading CrossEncoder reranker: {self.model_name}...")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                logger.warning(
                    f"Could not load CrossEncoder {self.model_name}: {exc}. "
                    f"Falling back to LexicalDensityReranker."
                )
                self._model = None
        return self._model

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            # Fall back to lexical density reranking
            fallback = LexicalDensityReranker()
            return fallback.rerank(query, candidates, top_k=top_k)

        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = model.predict(pairs)

        scored_candidates = []
        for cand, score in zip(candidates, scores):
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            scored_candidates.append(cand_copy)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]


class LexicalDensityReranker(BaseReranker):
    """
    Fast rule-based reranker that prioritizes exact industrial equipment tags,
    acronyms, and query term matches in candidate chunks.
    """

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        import re
        query_tags = set(re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?\b", query.upper()))
        scored_candidates = []

        for cand in candidates:
            cand_copy = dict(cand)
            text_upper = cand.get("text", "").upper()
            cand_tags = set(cand.get("tags", []))

            score = cand.get("score", 0.0) or 0.0

            # Boost if exact equipment tags match
            tag_matches = query_tags.intersection(cand_tags)
            score += len(tag_matches) * 0.25

            # Boost if query terms appear in text
            term_matches = sum(1 for q in query_tags if q in text_upper)
            score += term_matches * 0.05

            cand_copy["rerank_score"] = round(score, 4)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]


class MockReranker(BaseReranker):
    """Deterministic reranker for unit testing."""

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        scored = []
        for i, cand in enumerate(candidates):
            c = dict(cand)
            # Give higher scores to candidates containing query keywords
            boost = 1.0 if any(word.lower() in c.get("text", "").lower() for word in query.split()) else 0.0
            c["rerank_score"] = 1.0 - (i * 0.05) + boost
            scored.append(c)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


def get_reranker(engine: str = "lexical", model_name: Optional[str] = None) -> BaseReranker:
    """Factory to obtain the active reranker."""
    engine = engine.lower()
    if engine == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name or "BAAI/bge-reranker-v2-m3")
    elif engine == "mock":
        return MockReranker()
    else:
        return LexicalDensityReranker()
