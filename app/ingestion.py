"""
Document Ingestion Pipeline for SIH 26117.

Flow:
File -> File Type Detection -> Parser (Native Text / Local OCR Fallback)
-> Structured Records -> Semantic Chunking & Tag Extraction -> BGE-M3 Embeddings
-> Qdrant Vector Store -> SQLite Document Registry

Features:
- Incremental indexing, skip unchanged files, re-embed modified files, remove deleted files
- Resilient per-document error isolation (one corrupt file does not crash the pipeline)
- Comprehensive 8-metric summary reporting:
  (Indexed, Updated, Skipped, Deleted, Failed, Empty, OCR processed, OCR failed)
- Exposes clean APIs: ingest_document(path) and sync()
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bm25 import BM25Index
from .config import BM25_INDEX_PATH, DATA_DIR
from .chunking import chunk_documents
from .document_registry import DocumentRegistry
from .embeddings import get_embedding_provider
from .file_utils import calculate_file_hash, is_supported_file
from .parsers import parse_file
from .store import VectorStore

logger = logging.getLogger(__name__)


class IngestionEngine:

    def __init__(self, embedding_provider=None, vector_store=None, registry=None):
        logger.info("Initializing IngestionEngine...")

        self.embedder = embedding_provider or get_embedding_provider()
        vector_size = self.embedder.get_dimension()

        self.store = vector_store or VectorStore(vector_size=vector_size)
        self.store.ensure_collection()

        self.registry = registry or DocumentRegistry()
        self.bm25 = BM25Index()
        if BM25_INDEX_PATH.exists():
            try:
                self.bm25.load(BM25_INDEX_PATH)
            except Exception:
                pass
        self._all_indexed_chunks: List[Dict[str, Any]] = list(self.bm25.doc_payloads)

    @staticmethod
    def create_document_id(path: Path) -> str:
        """Create a stable, deterministic ID based on the normalized absolute path."""
        normalized_path = str(path.resolve()).lower()
        return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:32]

    def get_document_files(self, data_dir: Path = DATA_DIR) -> List[Path]:
        """Discover all supported document files in the knowledge base."""
        files = []
        for path in data_dir.rglob("*"):
            if path.is_file() and is_supported_file(path):
                files.append(path)
        return sorted(files)

    def delete_document_vectors(self, document_id: str):
        """Remove all Qdrant vectors belonging to a document and update BM25."""
        self.store.delete_by_document_id(document_id)
        self._all_indexed_chunks = [c for c in self._all_indexed_chunks if c.get("document_id") != document_id]
        self.bm25.index_chunks(self._all_indexed_chunks)
        try:
            self.bm25.save(BM25_INDEX_PATH)
        except Exception:
            pass

    def ingest_document(self, path: Path, force: bool = False) -> Dict[str, Any]:
        """
        Ingest or re-index a single document.

        Returns:
            Dict containing status ('indexed', 'updated', 'skipped', 'empty', 'failed'),
            chunk_count, ocr_used, document_id, error (if any).
        """
        path = Path(path).resolve()
        filename = path.name

        if not path.exists():
            return {"status": "failed", "filename": filename, "error": "File does not exist"}

        if not is_supported_file(path):
            return {"status": "failed", "filename": filename, "error": "Unsupported file format"}

        document_id = self.create_document_id(path)
        current_hash = calculate_file_hash(path)
        existing = self.registry.get_by_path(path)

        # Check if unchanged
        if existing and existing["file_hash"] == current_hash and not force:
            logger.info(f"SKIP: {filename} (unchanged)")
            return {
                "status": "skipped",
                "filename": filename,
                "document_id": document_id,
                "chunk_count": existing.get("chunk_count", 0),
                "ocr_used": existing.get("ocr_used", False),
            }

        # Check if modified
        is_update = existing is not None
        if is_update:
            logger.info(f"UPDATE: {filename} (removing existing vectors)...")
            self.delete_document_vectors(existing["document_id"])

        # Parse document
        logger.info(f"PARSING: {filename}...")
        try:
            records = parse_file(path)
        except Exception as error:
            logger.error(f"ERROR parsing {filename}: {error}")
            return {
                "status": "failed",
                "filename": filename,
                "document_id": document_id,
                "error": str(error),
            }

        if not records:
            logger.warning(f"EMPTY: No text or OCR content extracted from {filename}")
            return {
                "status": "empty",
                "filename": filename,
                "document_id": document_id,
            }

        # Chunk document
        chunks = chunk_documents(records)
        if not chunks:
            logger.warning(f"EMPTY: No chunks created for {filename}")
            return {
                "status": "empty",
                "filename": filename,
                "document_id": document_id,
            }

        # Inject document_id into every chunk and extract graph triples
        from .config import ENABLE_GRAPH_RAG

        for chunk in chunks:
            chunk["document_id"] = document_id

        if ENABLE_GRAPH_RAG:
            from .graph_rag import get_graph_rag

            graph_rag = get_graph_rag()

            for chunk in chunks:
                text_context = chunk.get("text", "")

                if len(text_context) > 100:
                    triples = graph_rag.extract_triples_from_text(
                        text_context
                    )

                    if triples:
                        graph_rag.add_triples(
                            triples,
                            source=filename
                        )

        # Determine OCR status
        any_ocr_used = any(c.get("ocr_used", False) for c in chunks)

        # Embed chunks
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"EMBEDDING: Generating vectors for {len(texts)} chunks of {filename}...")
        try:
            vectors = self.embedder.embed_documents(texts)
        except Exception as embed_err:
            logger.error(f"ERROR embedding {filename}: {embed_err}")
            return {
                "status": "failed",
                "filename": filename,
                "document_id": document_id,
                "error": str(embed_err),
            }

        # Upsert into Qdrant
        self.store.upsert(vectors, chunks, document_id=document_id)

        # Update BM25 lexical index
        if is_update:
            self._all_indexed_chunks = [c for c in self._all_indexed_chunks if c.get("document_id") != document_id]
        self._all_indexed_chunks.extend(chunks)
        self.bm25.index_chunks(self._all_indexed_chunks)
        try:
            self.bm25.save(BM25_INDEX_PATH)
        except Exception:
            pass

        # Extract metadata for registry
        first_chunk = chunks[0]
        file_type = first_chunk.get("file_type", path.suffix.lower().lstrip("."))
        category = first_chunk.get("category", "general")
        document_type = first_chunk.get("document_type", "general")

        # Update or add to SQLite registry
        if is_update:
            self.registry.update(
                document_id=document_id,
                file_hash=current_hash,
                file_type=file_type,
                category=category,
                document_type=document_type,
                status="indexed",
                ocr_used=any_ocr_used,
                chunk_count=len(chunks),
            )
        else:
            self.registry.add(
                document_id=document_id,
                filename=filename,
                path=path,
                file_hash=current_hash,
                file_type=file_type,
                category=category,
                document_type=document_type,
                status="indexed",
                ocr_used=any_ocr_used,
                chunk_count=len(chunks),
            )

        status_str = "updated" if is_update else "indexed"
        logger.info(f"INDEXED: {filename} ({len(chunks)} chunks, OCR: {any_ocr_used})")

        return {
            "status": status_str,
            "filename": filename,
            "document_id": document_id,
            "chunk_count": len(chunks),
            "ocr_used": any_ocr_used,
        }

    def remove_deleted_files(self, current_files: List[Path]) -> int:
        """Detect and clean up files deleted from disk."""
        current_paths = {str(p.resolve()) for p in current_files}
        registered_documents = self.registry.list_documents()

        removed_count = 0
        for doc in registered_documents:
            reg_path = str(Path(doc["path"]).resolve())
            if reg_path not in current_paths:
                logger.info(f"DELETE: Removing deleted file {doc['filename']}...")
                self.delete_document_vectors(doc["document_id"])
                self.registry.delete(doc["document_id"])
                removed_count += 1

        return removed_count

    def sync(self, data_dir: Path = DATA_DIR) -> Dict[str, int]:
        """
        Synchronize the knowledge base directory with Qdrant and SQLite registry.
        """
        print("=" * 70)
        print("SIH 26117 — SOVEREIGN KNOWLEDGE BASE SYNC")
        print("=" * 70)

        files = self.get_document_files(data_dir)
        print(f"Supported files found: {len(files)}\n")

        stats = {
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "failed": 0,
            "empty": 0,
            "ocr_processed": 0,
            "ocr_failed": 0,
        }

        for path in files:
            res = self.ingest_document(path)
            status = res["status"]

            if status == "indexed":
                stats["indexed"] += 1
                if res.get("ocr_used"):
                    stats["ocr_processed"] += 1
            elif status == "updated":
                stats["updated"] += 1
                if res.get("ocr_used"):
                    stats["ocr_processed"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "empty":
                stats["empty"] += 1
            elif status == "failed":
                stats["failed"] += 1
                if "ocr" in str(res.get("error", "")).lower():
                    stats["ocr_failed"] += 1

        stats["deleted"] = self.remove_deleted_files(files)

        print()
        print("=" * 70)
        print("KNOWLEDGE BASE SYNC COMPLETE")
        print("=" * 70)
        print(f"  New files indexed : {stats['indexed']}")
        print(f"  Updated files     : {stats['updated']}")
        print(f"  Skipped (unchanged: {stats['skipped']}")
        print(f"  Deleted files     : {stats['deleted']}")
        print(f"  Empty documents   : {stats['empty']}")
        print(f"  Failed documents  : {stats['failed']}")
        print(f"  OCR processed     : {stats['ocr_processed']}")
        print(f"  OCR failed        : {stats['ocr_failed']}")
        print("=" * 70)

        return stats

    def close(self):
        """Close vector store connection cleanly."""
        if hasattr(self.store, "close"):
            self.store.close()


def ingest_document(path: Path) -> Dict[str, Any]:
    """Clean module-level API for ingesting a single document."""
    engine = IngestionEngine()
    return engine.ingest_document(path)


def main():
    engine = IngestionEngine()
    engine.sync()


if __name__ == "__main__":
    main()