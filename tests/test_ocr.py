"""
Unit tests for Local OCR Pipeline and Text Quality Checking.
"""

import unittest
from pathlib import Path
from PIL import Image

from app.ocr import (
    BaseOCREngine,
    MockOCREngine,
    RapidOCREngine,
    TesseractOCREngine,
    extract_text_with_ocr,
    get_ocr_engine,
    is_text_insufficient,
)


class TestOCRPipeline(unittest.TestCase):

    def test_is_text_insufficient_empty(self):
        self.assertTrue(is_text_insufficient(""))
        self.assertTrue(is_text_insufficient(None))
        self.assertTrue(is_text_insufficient("   \n\t  "))

    def test_is_text_insufficient_short(self):
        # Default threshold is 50 chars
        self.assertTrue(is_text_insufficient("Too short"))
        self.assertTrue(is_text_insufficient("Page 1"))

    def test_is_text_insufficient_sufficient_text(self):
        rich_text = (
            "Standard Operating Procedure for Primary Coolant Loop Inspection. "
            "Verify pressure transmitter PT-101 reads within 15-25 bar range."
        )
        self.assertFalse(is_text_insufficient(rich_text))

    def test_mock_ocr_engine(self):
        engine = MockOCREngine(mock_text="Inspection Report PT-101 XV-301 Status: Passed")
        self.assertTrue(engine.is_available())

        # Create small test image
        img = Image.new("RGB", (100, 100), color="white")
        text, conf = engine.extract_text_from_image(img)

        self.assertIn("PT-101", text)
        self.assertIn("XV-301", text)
        self.assertGreaterEqual(conf, 0.9)

    def test_ocr_factory_mock(self):
        engine = get_ocr_engine("mock")
        self.assertIsInstance(engine, MockOCREngine)

    def test_technical_tag_preservation_in_ocr(self):
        # Verify technical strings maintain punctuation and casing
        sample_output = "Tag: PT-101, Flow: FT-204, Valve: XV-301, Power: 24 VDC"
        engine = MockOCREngine(mock_text=sample_output)
        img = Image.new("RGB", (50, 50), color="white")
        text, _ = engine.extract_text_from_image(img)

        self.assertIn("PT-101", text)
        self.assertIn("FT-204", text)
        self.assertIn("XV-301", text)
        self.assertIn("24 VDC", text)


if __name__ == "__main__":
    unittest.main()
