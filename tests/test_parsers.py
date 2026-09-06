"""
Unit tests for Document Parsers (PDF, DOCX, XLSX, and Image).
"""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from app.ocr import MockOCREngine
from app.parsers import parse_docx, parse_file, parse_image, parse_xlsx


class TestParsers(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_unsupported_file(self):
        unsupported = self.dir_path / "sample.xyz"
        unsupported.write_text("random content", encoding="utf-8")
        results = parse_file(unsupported)
        self.assertEqual(results, [])

    def test_parse_image(self):
        # Create a test image
        img_path = self.dir_path / "test_diagram.png"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(img_path)

        # Use mock OCR for testing
        from app import ocr
        original_engine = ocr._ACTIVE_OCR_ENGINE
        ocr._ACTIVE_OCR_ENGINE = MockOCREngine(mock_text="Tag: PT-101 Pressure: 24.5 bar")

        try:
            records = parse_image(img_path)
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertIn("PT-101", rec["text"])
            self.assertEqual(rec["location"], "Image")
            self.assertEqual(rec["file_type"], "png")
            self.assertEqual(rec["content_type"], "image")
            self.assertTrue(rec["ocr_used"])
            self.assertEqual(rec["extraction_method"], "ocr")
        finally:
            ocr._ACTIVE_OCR_ENGINE = original_engine

    def test_parse_docx(self):
        try:
            import docx
        except ImportError:
            self.skipTest("python-docx not installed yet")

        doc = docx.Document()
        doc.add_paragraph("Section 1: General Inspection Guidelines.")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Equipment"
        table.cell(0, 1).text = "Status"
        table.cell(1, 0).text = "PT-101"
        table.cell(1, 1).text = "Operational"

        docx_path = self.dir_path / "test_inspection.docx"
        doc.save(str(docx_path))

        records = parse_docx(docx_path)
        self.assertEqual(len(records), 1)
        text = records[0]["text"]
        self.assertIn("General Inspection Guidelines", text)
        self.assertIn("Equipment: PT-101", text)
        self.assertIn("Status: Operational", text)
        self.assertEqual(records[0]["file_type"], "docx")
        self.assertFalse(records[0]["ocr_used"])

    def test_parse_xlsx(self):
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl not installed yet")

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Instruments"
        ws.append(["Tag", "Type", "Range", "Power"])
        ws.append(["PT-101", "Pressure", "0-100 bar", "24 VDC"])
        ws.append(["FT-204", "Flow", "0-500 m3/h", "24 VDC"])

        xlsx_path = self.dir_path / "test_inventory.xlsx"
        wb.save(str(xlsx_path))

        records = parse_xlsx(xlsx_path)
        self.assertGreaterEqual(len(records), 2)
        all_text = " ".join(r["text"] for r in records)
        self.assertIn("PT-101", all_text)
        self.assertIn("FT-204", all_text)
        self.assertIn("24 VDC", all_text)
        self.assertEqual(records[0]["file_type"], "xlsx")
        self.assertIn("Sheet Instruments", records[0]["location"])


if __name__ == "__main__":
    unittest.main()
