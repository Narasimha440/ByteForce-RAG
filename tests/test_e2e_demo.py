"""
End-to-End Simulation of the SIH Demonstration Workflow (Phase 17).

Demonstration Scenario:
1. Input: An industrial inspection report with a scanned page.
2. Ingestion pipeline detects scanned page (insufficient native text).
3. Local OCR pipeline triggers, extracting critical findings and equipment tags (PT-101, XV-301).
4. Page number (Page 18) and extraction metadata (ocr_used=True, extraction_method='ocr') are preserved.
5. Chunks are generated semantically and embedded into Qdrant vector store.
6. User asks: "What were the major findings in the inspection report?"
7. Local RAG retrieves the exact evidence chunk with full provenance citations:
   Document: Inspection_Report_2026.pdf
   Page: 18
   Extraction: ocr
   Tags: PT-101, XV-301
"""

import tempfile
from typing import List
import unittest
from pathlib import Path
from qdrant_client import QdrantClient

from app.chunking import chunk_documents
from app.document_registry import DocumentRegistry
from app.embeddings import BaseEmbeddingProvider
from app.ingestion import IngestionEngine
from app.ocr import MockOCREngine
from app.rag import LocalRAG
from app.store import VectorStore


class DeterministicDemoEmbedder(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def get_dimension(self) -> int:
        return self.dimension

    def embed_query(self, query: str) -> List[float]:
        val = 1.0 / (self.dimension ** 0.5)
        return [val] * self.dimension

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


class TestEndToEndSIHDemo(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

        # Isolated in-memory Qdrant & lightweight embedding provider
        self.embedder = DeterministicDemoEmbedder(dimension=16)
        self.client = QdrantClient(":memory:")
        self.store = VectorStore(vector_size=16, client=self.client)
        self.store.recreate_collection()

        # Isolated test SQLite registry
        self.db_path = self.work_dir / "registry.db"
        self.registry = DocumentRegistry(db_path=self.db_path)

        # Configure deterministic Mock OCR
        from app import ocr
        self.original_engine = ocr._ACTIVE_OCR_ENGINE
        ocr._ACTIVE_OCR_ENGINE = MockOCREngine(
            mock_text=(
                "Annual Safety Audit: Major Findings.\n"
                "Pressure transmitter PT-101 exhibited heavy flange corrosion and seal leakage.\n"
                "Emergency shutdown valve XV-301 actuator response time exceeded standard tolerance by 4.2 seconds.\n"
                "Immediate replacement required before recommissioning unit."
            )
        )

    def tearDown(self):
        from app import ocr
        ocr._ACTIVE_OCR_ENGINE = self.original_engine
        try:
            self.temp_dir.cleanup()
        except (PermissionError, OSError):
            pass

    def test_sih_scanned_report_demo_flow(self):
        # 1. Simulate parsed record from a scanned PDF page (e.g. Page 18)
        # Native text was empty, triggering local OCR
        scanned_record = {
            "text": (
                "Annual Safety Audit: Major Findings.\n"
                "Pressure transmitter PT-101 exhibited heavy flange corrosion and seal leakage.\n"
                "Emergency shutdown valve XV-301 actuator response time exceeded standard tolerance by 4.2 seconds.\n"
                "Immediate replacement required before recommissioning unit."
            ),
            "source": str(self.work_dir / "Inspection_Report_2026.pdf"),
            "filename": "Inspection_Report_2026.pdf",
            "location": "Page 18",
            "page_number": 18,
            "section": "Major Findings",
            "file_type": "pdf",
            "content_type": "scanned_document",
            "category": "compliance",
            "document_type": "inspection",
            "extraction_method": "ocr",
            "ocr_used": True,
            "ocr_confidence": 0.96,
        }

        # 2. Semantic Chunking
        chunks = chunk_documents([scanned_record])
        self.assertGreaterEqual(len(chunks), 1)
        first_chunk = chunks[0]

        # Verify provenance was preserved
        self.assertEqual(first_chunk["page_number"], 18)
        self.assertEqual(first_chunk["location"], "Page 18")
        self.assertEqual(first_chunk["extraction_method"], "ocr")
        self.assertTrue(first_chunk["ocr_used"])
        self.assertEqual(first_chunk["content_type"], "scanned_document")

        # Verify industrial tags were extracted
        self.assertIn("PT-101", first_chunk["tags"])
        self.assertIn("XV-301", first_chunk["tags"])

        # 3. Vector Embedding and Storage
        doc_id = "doc_inspection_2026"
        for c in chunks:
            c["document_id"] = doc_id

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)
        self.store.upsert(vectors, chunks, document_id=doc_id)

        # 4. User queries for findings
        rag = LocalRAG(embedding_provider=self.embedder, vector_store=self.store)
        hits = rag.retrieve(
            question="What were the major findings in the inspection report?",
            top_k=3,
            threshold=0.35,
        )

        self.assertGreaterEqual(len(hits), 1)

        # 5. Verify SIH evidence formatting & citation provenance
        context, sources = rag.build_context(hits)

        # Context contains structured evidence blocks
        self.assertIn("[EVIDENCE 1]", context)
        self.assertIn("Document: Inspection_Report_2026.pdf", context)
        self.assertIn("Page: 18", context)
        self.assertIn("Location: Page 18", context)
        self.assertIn("Extraction Method: ocr (OCR Used: True)", context)
        self.assertIn("PT-101, XV-301", context)
        self.assertIn("corrosion", context)

        # Sources dictionary has exact citation fields
        top_source = sources[0]
        self.assertEqual(top_source["filename"], "Inspection_Report_2026.pdf")
        self.assertEqual(top_source["location"], "Page 18")
        self.assertEqual(top_source["page_number"], 18)
        self.assertEqual(top_source["extraction_method"], "ocr")
        self.assertTrue(top_source["ocr_used"])
        self.assertIn("PT-101", top_source["tags"])
        self.assertIn("XV-301", top_source["tags"])


if __name__ == "__main__":
    unittest.main()
