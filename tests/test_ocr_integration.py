"""
Integration tests for Real Local OCR (RapidOCR) and PDF page rendering (PyMuPDF).
"""

import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from app.parsers import parse_file, parse_image, parse_pdf
from app.ocr import get_ocr_engine, RapidOCREngine, render_pdf_page_to_image


class TestOCRIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except (PermissionError, OSError):
            pass

    def test_real_rapidocr_engine_available(self):
        engine = get_ocr_engine("rapidocr")
        self.assertTrue(engine.is_available(), "RapidOCR should be available in the local environment")

    def test_real_rapidocr_image_extraction(self):
        # Create an image with clear high-contrast text
        img_path = self.work_dir / "test_panel_tag.png"
        img = Image.new("RGB", (500, 120), color="white")
        draw = ImageDraw.Draw(img)
        # Draw clean black text
        draw.text((20, 30), "VALVE XV-301 STATUS 24 VDC", fill="black")
        img.save(img_path)

        records = parse_image(img_path)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertTrue(rec["ocr_used"])
        self.assertEqual(rec["extraction_method"], "ocr")
        self.assertGreater(rec["ocr_confidence"], 0.0)

        # Verify key industrial tags extracted
        text = rec["text"].upper()
        self.assertTrue("XV-301" in text or "301" in text or "VALVE" in text)

    def test_pdf_with_native_and_scanned_pages(self):
        import pymupdf

        pdf_path = self.work_dir / "hybrid_inspection_report.pdf"
        doc = pymupdf.open()

        # Page 1: Native digital text
        page1 = doc.new_page()
        page1.insert_text(
            (50, 72),
            "General Specification: Instrument Control Panel ICP-01.\n"
            "All power circuits shall operate on 120 VAC nominal.\n"
            "Pressure transmitters PT-101 and flow meters FT-204 calibrated.",
            fontsize=12,
        )

        # Page 2: Image-only page (simulating scanned document)
        img = Image.new("RGB", (600, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 40), "SCANNED FINDINGS: CRITICAL CORROSION ON PUMP P-101A", fill="black")
        img_tmp = self.work_dir / "scanned_page.png"
        img.save(img_tmp)

        page2 = doc.new_page(width=600, height=800)
        # Insert image into page rect without any native text
        page2.insert_image(pymupdf.Rect(50, 50, 550, 250), filename=str(img_tmp))

        doc.save(str(pdf_path))
        doc.close()

        # Parse hybrid PDF
        records = parse_pdf(pdf_path)
        self.assertEqual(len(records), 2, "Both native and scanned pages must be extracted")

        # Page 1 must be native_text
        self.assertEqual(records[0]["page_number"], 1)
        self.assertEqual(records[0]["extraction_method"], "native_text")
        self.assertFalse(records[0]["ocr_used"])
        self.assertIn("ICP-01", records[0]["text"])

        # Page 2 must be OCR
        self.assertEqual(records[1]["page_number"], 2)
        self.assertEqual(records[1]["extraction_method"], "ocr")
        self.assertTrue(records[1]["ocr_used"])
        self.assertEqual(records[1]["content_type"], "scanned_document")


if __name__ == "__main__":
    unittest.main()
