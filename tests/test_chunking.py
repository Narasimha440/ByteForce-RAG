"""
Unit tests for Semantic Chunking, Provenance Preservation, and Tag Extraction.
"""

import unittest
from pathlib import Path

from app.chunking import chunk_documents, split_text_semantically
from app.document_metadata import extract_tags


class TestChunking(unittest.TestCase):

    def test_split_text_semantically_paragraphs(self):
        text = (
            "Section 1: Initial Calibration.\n"
            "Calibrate pressure transmitter PT-101 according to standard protocol.\n\n"
            "Section 2: Secondary Checks.\n"
            "Inspect safety valve XV-301 and ensure flow transmitter FT-204 is connected."
        )
        chunks = split_text_semantically(text, chunk_size=200, chunk_overlap=40)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            # Ensure sentences are not sliced mid-word
            self.assertFalse(chunk.startswith(" "))
            self.assertFalse(chunk.endswith(" "))

    def test_chunk_documents_unique_ids_across_pages(self):
        records = [
            {
                "text": "Page one text detailing equipment PT-101 and pressure metrics.",
                "source": "C:/docs/inspection_report.pdf",
                "filename": "inspection_report.pdf",
                "location": "Page 1",
                "page_number": 1,
                "file_type": "pdf",
                "content_type": "document",
                "extraction_method": "native_text",
                "ocr_used": False,
            },
            {
                "text": "Page two text detailing inspection findings for valve XV-301.",
                "source": "C:/docs/inspection_report.pdf",
                "filename": "inspection_report.pdf",
                "location": "Page 2",
                "page_number": 2,
                "file_type": "pdf",
                "content_type": "scanned_document",
                "extraction_method": "ocr",
                "ocr_used": True,
                "ocr_confidence": 0.94,
            },
        ]

        chunks = chunk_documents(records)
        self.assertEqual(len(chunks), 2)

        # Verify chunk IDs are distinct across pages! (Fixes previous bug)
        self.assertNotEqual(chunks[0]["chunk_id"], chunks[1]["chunk_id"])
        self.assertEqual(chunks[0]["chunk_id"], "p1_c0")
        self.assertEqual(chunks[1]["chunk_id"], "p2_c0")

        # Verify provenance retention
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[0]["extraction_method"], "native_text")
        self.assertFalse(chunks[0]["ocr_used"])

        self.assertEqual(chunks[1]["page_number"], 2)
        self.assertEqual(chunks[1]["extraction_method"], "ocr")
        self.assertTrue(chunks[1]["ocr_used"])
        self.assertEqual(chunks[1]["content_type"], "scanned_document")

    def test_tag_extraction(self):
        text = (
            "Verify loop controller PLC-01 and transmitter PT-101. "
            "Also check FT-204 and emergency valve XV-301 with 24 VDC power supply. "
            "Refer to drawing P&ID legend."
        )
        tags = extract_tags(text)
        self.assertIn("PT-101", tags)
        self.assertIn("FT-204", tags)
        self.assertIn("XV-301", tags)
        self.assertIn("24 VDC", tags)
        self.assertIn("P&ID", tags)


if __name__ == "__main__":
    unittest.main()
