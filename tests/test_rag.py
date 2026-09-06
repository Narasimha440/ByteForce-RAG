"""
Unit tests for LocalRAG (Retrieval, Threshold Filtering, and Provenance Citation).
"""

import unittest
from unittest.mock import MagicMock
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from app.embeddings import MockEmbeddingProvider
from app.rag import LocalRAG, INSUFFICIENT_KNOWLEDGE_MESSAGE
from app.store import VectorStore


class TestLocalRAG(unittest.TestCase):

    def setUp(self):
        self.embedder = MockEmbeddingProvider(dimension=8)
        self.client = QdrantClient(":memory:")
        self.store = VectorStore(vector_size=8, client=self.client)
        self.store.recreate_collection()
        self.rag = LocalRAG(embedding_provider=self.embedder, vector_store=self.store)

    def test_build_context_evidence_provenance(self):
        hit = ScoredPoint(
            id="test-point-id",
            version=1,
            score=0.92,
            payload={
                "filename": "Inspection_Report.pdf",
                "source": "C:/docs/Inspection_Report.pdf",
                "location": "Page 18",
                "page_number": 18,
                "section": "Major Findings",
                "category": "compliance",
                "document_type": "inspection",
                "extraction_method": "ocr",
                "ocr_used": True,
                "tags": ["PT-101", "XV-301"],
                "chunk_id": "p18_c0",
                "text": "Critical corrosion observed on pressure transmitter PT-101 flange.",
            },
            vector=None,
        )

        context, sources = self.rag.build_context([hit])

        # Check evidence citation markers
        self.assertIn("[EVIDENCE 1]", context)
        self.assertIn("Document: Inspection_Report.pdf", context)
        self.assertIn("Page: 18", context)
        self.assertIn("Location: Page 18", context)
        self.assertIn("Section: Major Findings", context)
        self.assertIn("Extraction Method: ocr (OCR Used: True)", context)
        self.assertIn("PT-101, XV-301", context)

        # Check structured sources metadata
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["filename"], "Inspection_Report.pdf")
        self.assertEqual(sources[0]["page_number"], 18)
        self.assertEqual(sources[0]["score"], 0.92)
        self.assertTrue(sources[0]["ocr_used"])
        self.assertEqual(sources[0]["extraction_method"], "ocr")

    def test_similarity_threshold_filtering(self):
        # Insert a chunk
        vec = self.embedder.embed_query("Normal operational procedure")
        chunks = [{
            "chunk_id": "c1",
            "text": "Standard routine maintenance protocol.",
            "category": "standards",
        }]
        self.store.upsert([vec], chunks, document_id="doc_thresh_1")

        # Set an impossibly high threshold (e.g. 0.9999) -> must filter out
        hits = self.rag.retrieve("Unrelated query about emergency valves", threshold=0.9999)
        self.assertEqual(len(hits), 0)

    def test_insufficient_knowledge_fallback(self):
        # If knowledge base has no hits exceeding threshold
        answer, sources = self.rag.answer("What is the temperature limit for boiler B-99?", threshold=0.95)
        self.assertEqual(answer, INSUFFICIENT_KNOWLEDGE_MESSAGE)
        self.assertEqual(len(sources), 0)


if __name__ == "__main__":
    unittest.main()
