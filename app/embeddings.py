"""
Modular Embedding Provider Interface for SIH 26117.

Supports:
- Local BAAI/bge-m3 embedding via SentenceTransformers
- Pluggable APIEmbeddingProvider for moving model inference to Nandan's dedicated laptop
- MockEmbeddingProvider for deterministic unit testing
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Optional

import requests

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_API_URL,
)

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract interface for document and query embeddings."""

    @abstractmethod
    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed a list of document chunk texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query string."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimensionality (e.g. 1024 for BAAI/bge-m3)."""
        pass


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local in-process embedding provider using BAAI/bge-m3.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._dimension = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading local embedding model: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._get_model()
        return self._dimension

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 32,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode(query, normalize_embeddings=True)
        return embedding.tolist()


class APIEmbeddingProvider(BaseEmbeddingProvider):
    """
    HTTP API client embedding provider.

    Enables offloading embedding generation to Nandan's dedicated model server.
    Expected endpoints:
      POST /embed (JSON: {"texts": [...]}) -> {"embeddings": [[...]]}
      GET /dimension -> {"dimension": 1024}
    """

    def __init__(self, api_url: str = EMBEDDING_API_URL, dimension: int = 1024):
        self.api_url = api_url.rstrip("/")
        self.dimension = dimension

    def get_dimension(self) -> int:
        try:
            resp = requests.get(f"{self.api_url}/dimension", timeout=5)
            if resp.ok:
                return resp.json().get("dimension", self.dimension)
        except Exception:
            pass
        return self.dimension

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        resp = requests.post(
            f"{self.api_url}/embed",
            json={"texts": texts, "batch_size": batch_size},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_query(self, query: str) -> List[float]:
        resp = requests.post(
            f"{self.api_url}/embed_query",
            json={"query": query},
            timeout=30,
        )
        if resp.ok:
            return resp.json()["embedding"]
        # Fallback to general /embed
        return self.embed_documents([query])[0]


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, lightweight embedding provider for unit tests."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def get_dimension(self) -> int:
        return self.dimension

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(query.encode("utf-8")).digest()
        # Produce a normalized pseudo-vector from hash
        raw = [(b / 255.0) - 0.5 for b in h[:self.dimension]]
        norm = sum(x ** 2 for x in raw) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in raw]


_ACTIVE_EMBEDDING_PROVIDER: Optional[BaseEmbeddingProvider] = None


def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Factory to retrieve or create the active embedding provider.
    """
    global _ACTIVE_EMBEDDING_PROVIDER

    ptype = (provider_type or EMBEDDING_PROVIDER).lower()

    if _ACTIVE_EMBEDDING_PROVIDER is not None and not provider_type:
        return _ACTIVE_EMBEDDING_PROVIDER

    provider: BaseEmbeddingProvider

    if ptype == "mock":
        provider = MockEmbeddingProvider()
    elif ptype == "api":
        provider = APIEmbeddingProvider()
    else:  # "local" (default)
        provider = LocalEmbeddingProvider()

    if not provider_type:
        _ACTIVE_EMBEDDING_PROVIDER = provider

    return provider
