"""
Integration tests for Hybrid Retrieval (Dense BGE-M3 + BM25) and Reranking.
"""

import unittest
from qdrant_client import QdrantClient

from app.bm25 import BM25Index
from app.embeddings import MockEmbeddingProvider
from app.rag import LocalRAG
from app.reranker import LexicalDensityReranker, MockReranker
from app.store import VectorStore


class TestHybridRAG(unittest.TestCase):

    def setUp(self):
        self.embedder = MockEmbeddingProvider(dimension=16)
        self.client = QdrantClient(":memory:")
        self.store = VectorStore(vector_size=16, client=self.client)
        self.store.recreate_collection()

        self.chunks = [
            {
                "chunk_id": "c_cdu",
                "text": "Crude Distillation Unit (CDU) atmospheric tower T-101 operates at 360 C.",
                "filename": "cdu_sop.docx",
                "location": "Page 3",
                "page_number": 3,
                "category": "engineering",
                "document_type": "standard",
                "tags": ["CDU", "T-101"],
            },
            {
                "chunk_id": "c_esd",
                "text": "Emergency Shutdown Valve XV-301 actuator trip test and proof testing per SIL-2 requirements.",
                "filename": "esd_manual.pdf",
                "location": "Page 12",
                "page_number": 12,
                "category": "compliance",
                "document_type": "inspection",
                "tags": ["XV-301", "SIL-2", "ESD"],
            },
            {
                "chunk_id": "c_fccu",
                "text": "Fluid Catalytic Cracking Unit (FCCU) catalyst regeneration cyclone differential pressure monitoring.",
                "filename": "fccu_manual.docx",
                "location": "Page 7",
                "page_number": 7,
                "category": "engineering",
                "document_type": "standard",
                "tags": ["FCCU", "DP-401"],
            },
        ]

        # Populate Qdrant
        vectors = self.embedder.embed_documents([c["text"] for c in self.chunks])
        self.store.upsert(vectors, self.chunks, document_id="doc_refinery_test")

        # Populate BM25
        self.bm25 = BM25Index()
        self.bm25.index_chunks(self.chunks)

        self.rag = LocalRAG(
            embedding_provider=self.embedder,
            vector_store=self.store,
            bm25_index=self.bm25,
            reranker=LexicalDensityReranker(),
        )

    def test_hybrid_exact_equipment_tag_retrieval(self):
        # Querying specifically for the emergency valve XV-301
        hits = self.rag.retrieve("What are the SIL-2 requirements for XV-301?", top_k=2, threshold=0.0)
        self.assertGreaterEqual(len(hits), 1)

        top_hit = hits[0]
        self.assertEqual(top_hit["chunk_id"], "c_esd")
        self.assertIn("XV-301", top_hit["tags"])
        self.assertIn("SIL-2", top_hit["tags"])

    def test_reranker_boosts_relevant_document(self):
        reranker = MockReranker()
        candidates = [
            {"text": "Generic refinery pump guidelines", "tags": []},
            {"text": "FCCU catalyst regeneration cyclone differential pressure", "tags": ["FCCU"]},
        ]
        reranked = reranker.rerank("FCCU catalyst", candidates, top_k=2)
        self.assertIn("FCCU", reranked[0]["text"])


if __name__ == "__main__":
    unittest.main()
