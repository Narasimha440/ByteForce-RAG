"""
Vector Store Management with Qdrant for SIH 26117.

Supports:
- Connection to local Qdrant server (http://localhost:6333)
- Automatic graceful fallback to embedded on-disk Qdrant storage (data/qdrant_storage)
  when Docker is offline or not installed
- Extended Qdrant payload preserving all industrial provenance fields
- Deterministic, collision-free UUID5 point IDs
- Metadata-aware filtering (category, document_type, tags, etc.)
- Atomic document deletion and re-indexing
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import (
    COLLECTION_NAME,
    QDRANT_PREFER_LOCAL,
    QDRANT_STORAGE_PATH,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self, vector_size: int, client: Optional[QdrantClient] = None):
        self.vector_size = vector_size

        if client is not None:
            self.client = client
        elif QDRANT_PREFER_LOCAL:
            logger.info(f"Using embedded local Qdrant at {QDRANT_STORAGE_PATH}")
            QDRANT_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
        else:
            try:
                # Attempt connecting to server
                test_client = QdrantClient(url=QDRANT_URL, timeout=3.0)
                test_client.get_collections()
                self.client = test_client
                logger.info(f"Connected to Qdrant server at {QDRANT_URL}")
            except Exception as exc:
                logger.warning(
                    f"Could not connect to Qdrant at {QDRANT_URL} ({exc}). "
                    f"Falling back to embedded on-disk storage at {QDRANT_STORAGE_PATH}."
                )
                QDRANT_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(QDRANT_STORAGE_PATH))

    def recreate_collection(self):
        """Drop and recreate the knowledge collection."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def ensure_collection(self):
        """Ensure the collection exists with correct dimension and distance."""
        try:
            collections = self.client.get_collections()
            existing_names = {c.name for c in collections.collections}
        except Exception:
            existing_names = set()

        if COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        vectors: List[List[float]],
        chunks: List[Dict[str, Any]],
        document_id: str,
        batch_size: int = 64,
    ):
        """
        Upsert vectors and extended payload into Qdrant in batches.
        Uses collision-free deterministic UUID5 point IDs.
        """
        total = len(vectors)
        if total == 0:
            return

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_vectors = vectors[start:end]
            batch_chunks = chunks[start:end]

            points: List[PointStruct] = []

            for offset, (vector, chunk) in enumerate(
                zip(batch_vectors, batch_chunks)
            ):
                chunk_id = chunk.get("chunk_id", f"idx_{start + offset}")

                # Deterministic collision-free point ID
                point_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"sih26117:{document_id}:{chunk_id}",
                    )
                )

                # Full industrial provenance payload per Phase 11
                payload = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", "Unknown"),
                    "filename": chunk.get("filename", "Unknown"),
                    "location": chunk.get("location", "Unknown"),
                    "page_number": chunk.get("page_number", 1),
                    "section": chunk.get("section", None),
                    "file_type": chunk.get("file_type", "unknown"),
                    "content_type": chunk.get("content_type", "document"),
                    "category": chunk.get("category", "general"),
                    "document_type": chunk.get("document_type", "general"),
                    "tags": chunk.get("tags", []),
                    "ocr_used": chunk.get("ocr_used", False),
                    "extraction_method": chunk.get("extraction_method", "native_text"),
                    "ocr_confidence": chunk.get("ocr_confidence", None),
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

        logger.info(f"Upserted {total} vectors for document {document_id}")

    def delete_by_document_id(self, document_id: str):
        """Remove all vectors belonging to a document from Qdrant."""
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted vectors for document: {document_id}")
        except Exception as exc:
            logger.warning(f"Error deleting vectors for {document_id}: {exc}")

    def search(
        self,
        vector: List[float],
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Search for most similar points with optional metadata filtering.
        """
        query_filter = None
        if filter_dict:
            conditions = []
            for key, val in filter_dict.items():
                if isinstance(val, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=val))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=val))
                    )
            if conditions:
                query_filter = Filter(must=conditions)

        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        return result.points

    def close(self):
        """Close client connection cleanly."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass