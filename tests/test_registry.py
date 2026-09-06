"""
Unit tests for DocumentRegistry (SQLite persistent tracking).
"""

import tempfile
import unittest
from pathlib import Path

from app.document_registry import DocumentRegistry


class TestDocumentRegistry(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_registry.db"
        self.registry = DocumentRegistry(db_path=self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except (PermissionError, OSError):
            pass

    def test_add_and_get_by_path(self):
        path = Path("C:/data/inspection_report.pdf")
        self.registry.add(
            document_id="doc_123",
            filename="inspection_report.pdf",
            path=path,
            file_hash="hash_abc",
            file_type="pdf",
            category="compliance",
            document_type="inspection",
            status="indexed",
            ocr_used=True,
            chunk_count=15,
        )

        doc = self.registry.get_by_path(path)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["document_id"], "doc_123")
        self.assertEqual(doc["file_hash"], "hash_abc")
        self.assertEqual(doc["category"], "compliance")
        self.assertEqual(doc["document_type"], "inspection")
        self.assertTrue(doc["ocr_used"])
        self.assertEqual(doc["chunk_count"], 15)

    def test_update_modified_document(self):
        path = Path("C:/data/specs.docx")
        self.registry.add(
            document_id="doc_456",
            filename="specs.docx",
            path=path,
            file_hash="hash_old",
            file_type="docx",
            category="standards",
            document_type="standard",
            status="indexed",
        )

        # Update file hash and chunk count
        self.registry.update(
            document_id="doc_456",
            file_hash="hash_new",
            chunk_count=20,
            ocr_used=False,
        )

        doc = self.registry.get_by_id("doc_456")
        self.assertEqual(doc["file_hash"], "hash_new")
        self.assertEqual(doc["chunk_count"], 20)

    def test_delete_document(self):
        path = Path("C:/data/temp.xlsx")
        self.registry.add(
            document_id="doc_789",
            filename="temp.xlsx",
            path=path,
            file_hash="hash_xyz",
            file_type="xlsx",
            category="templates",
            document_type="template",
        )

        self.assertIsNotNone(self.registry.get_by_id("doc_789"))
        self.registry.delete("doc_789")
        self.assertIsNone(self.registry.get_by_id("doc_789"))


if __name__ == "__main__":
    unittest.main()
