"""
Sovereign RAG Subsystem for SIH 26117 (Harsha).

100% local, air-gapped Retrieval-Augmented Generation Engine for industrial refinery operations.
Combines:
- Dense semantic vector search: BAAI/bge-m3 (1024-dim) via Qdrant
- Sparse lexical search: Okapi BM25 with industrial tag preservation
- Reciprocal Rank Fusion (RRF, k=60)
- Cross-Encoder deep attention reranking: BAAI/bge-reranker-v2-m3
- MRPL Domain Ontology & Equipment Knowledge Graph (XV-301, BDV-701, PT-101, FT-204)
- Anti-hallucination similarity thresholding (0.35) & numerical grounding verification
- Strict Agent JSON tool-calling interface
"""

from app.rag import (
    INSUFFICIENT_KNOWLEDGE_MESSAGE,
    LocalRAG,
    build_context,
    get_visual_evidence,
    retrieve,
    verify_numerical_grounding,
)
from app.bm25 import BM25Index, tokenize_industrial_text
from app.chunking import chunk_documents, split_text_semantically
from app.domain_expansion import expand_refinery_query, extract_query_equipment_tags
from app.embeddings import BaseEmbeddingProvider, LocalEmbeddingProvider, get_embedding_provider
from app.equipment_graph import get_equipment_dossier, format_dossier_box
from app.ingestion import IngestionEngine, ingest_document
from app.reranker import BaseReranker, CrossEncoderReranker, LexicalDensityReranker, get_reranker
from app.store import VectorStore

__all__ = [
    "INSUFFICIENT_KNOWLEDGE_MESSAGE",
    "LocalRAG",
    "retrieve",
    "build_context",
    "get_visual_evidence",
    "verify_numerical_grounding",
    "BM25Index",
    "tokenize_industrial_text",
    "chunk_documents",
    "split_text_semantically",
    "expand_refinery_query",
    "extract_query_equipment_tags",
    "BaseEmbeddingProvider",
    "LocalEmbeddingProvider",
    "get_embedding_provider",
    "get_equipment_dossier",
    "format_dossier_box",
    "IngestionEngine",
    "ingest_document",
    "BaseReranker",
    "CrossEncoderReranker",
    "LexicalDensityReranker",
    "get_reranker",
    "VectorStore",
]
