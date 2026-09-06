"""
Unit tests for BM25 Lexical Index and Industrial Tokenization.
"""

import tempfile
import unittest
from pathlib import Path

from app.bm25 import BM25Index, tokenize_industrial_text


class TestBM25Index(unittest.TestCase):

    def test_tokenize_industrial_text(self):
        text = "Calibrate PT-101 and flow meter FT-204 on 24 VDC circuit."
        tokens = tokenize_industrial_text(text)
        self.assertIn("pt-101", tokens)
        self.assertIn("ft-204", tokens)
        self.assertIn("24", tokens)
        self.assertIn("vdc", tokens)

    def test_bm25_search_exact_tags(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "Cooling water pump P-101A routine inspection checklist.",
                "tags": ["P-101A"],
                "filename": "checklist.pdf",
            },
            {
                "chunk_id": "c2",
                "text": "Fluid catalytic cracker regenerator temperature safety limits and cyclone delta-P.",
                "tags": ["FCCU", "TI-201"],
                "filename": "fccu_manual.docx",
            },
            {
                "chunk_id": "c3",
                "text": "Emergency shutdown valve XV-301 actuator calibration on 120 VAC solenoid.",
                "tags": ["XV-301", "120 VAC"],
                "filename": "esd_spec.docx",
            },
        ]

        index = BM25Index()
        index.index_chunks(chunks)

        # Query for exact tag
        results = index.search("XV-301", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        top_doc, score = results[0]
        self.assertEqual(top_doc["chunk_id"], "c3")
        self.assertGreater(score, 0.0)

    def test_bm25_save_and_load(self):
        chunks = [
            {
                "chunk_id": "c10",
                "text": "Crude distillation unit CDU preheat train exchanger E-101 inspection.",
                "tags": ["CDU", "E-101"],
            }
        ]
        index = BM25Index()
        index.index_chunks(chunks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "bm25.json"
            index.save(file_path)

            loaded_index = BM25Index()
            loaded_index.load(file_path)

            self.assertEqual(loaded_index.corpus_size, 1)
            results = loaded_index.search("E-101", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0]["chunk_id"], "c10")


if __name__ == "__main__":
    unittest.main()
