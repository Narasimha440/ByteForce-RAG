"""
Unit tests for VectorStore (Qdrant storage, payload, and search).
"""

import unittest
from qdrant_client import QdrantClient

from app.store import VectorStore


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        # Use an isolated in-memory Qdrant instance for fast, clean unit testing
        self.client = QdrantClient(":memory:")
        self.vector_size = 8
        self.store = VectorStore(vector_size=self.vector_size, client=self.client)
        self.store.recreate_collection()

    def test_upsert_and_payload_preservation(self):
        # Distinct orthogonal vectors to ensure distinct cosine similarities
        vec1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        chunks = [
            {
                "chunk_id": "p1_c0",
                "text": "Cooling loop inspection results for pump P-101A.",
                "source": "C:/docs/inspection_report.pdf",
                "filename": "inspection_report.pdf",
                "location": "Page 1",
                "page_number": 1,
                "section": "Findings",
                "file_type": "pdf",
                "content_type": "scanned_document",
                "category": "compliance",
                "document_type": "inspection",
                "tags": ["P-101A", "MAH"],
                "ocr_used": True,
                "extraction_method": "ocr",
            },
            {
                "chunk_id": "p2_c0",
                "text": "Control panel wiring standard for 24 VDC systems.",
                "source": "C:/docs/standard.docx",
                "filename": "standard.docx",
                "location": "Page 2",
                "page_number": 2,
                "section": "Power",
                "file_type": "docx",
                "content_type": "document",
                "category": "engineering",
                "document_type": "control_panel",
                "tags": ["24 VDC", "ICP"],
                "ocr_used": False,
                "extraction_method": "native_text",
            },
        ]

        self.store.upsert([vec1, vec2], chunks, document_id="doc_test_1")

        # Search with vec1 -> vec1 should rank #1
        results = self.store.search(vector=vec1, limit=5)
        self.assertEqual(len(results), 2)

        # Check payload attributes on top result
        top_hit = results[0]
        payload = top_hit.payload
        self.assertEqual(payload["document_id"], "doc_test_1")
        self.assertEqual(payload["chunk_id"], "p1_c0")
        self.assertEqual(payload["page_number"], 1)
        self.assertTrue(payload["ocr_used"])
        self.assertEqual(payload["extraction_method"], "ocr")
        self.assertIn("P-101A", payload["tags"])

    def test_search_with_metadata_filter(self):
        vec1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        chunks = [
            {
                "chunk_id": "p1_c0",
                "text": "Inspection document content.",
                "category": "compliance",
                "document_type": "inspection",
            },
            {
                "chunk_id": "p2_c0",
                "text": "Engineering P&ID content.",
                "category": "engineering",
                "document_type": "pid",
            },
        ]
        self.store.upsert([vec1, vec2], chunks, document_id="doc_filter_test")

        # Filter specifically for compliance
        results = self.store.search(
            vector=vec1,
            limit=5,
            filter_dict={"category": "compliance"},
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload["category"], "compliance")

    def test_delete_by_document_id(self):
        vec = [1.0] + [0.0] * (self.vector_size - 1)
        chunks = [{"chunk_id": "c1", "text": "Temporary document."}]

        self.store.upsert([vec], chunks, document_id="doc_to_delete")
        self.assertEqual(len(self.store.search(vec)), 1)

        self.store.delete_by_document_id("doc_to_delete")
        self.assertEqual(len(self.store.search(vec)), 0)


if __name__ == "__main__":
    unittest.main()
