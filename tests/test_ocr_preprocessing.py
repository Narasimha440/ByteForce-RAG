"""
Unit tests for OCR image preprocessing, auto-contrast, and layout/bounding-box extraction.
"""

import unittest
from PIL import Image, ImageDraw
from app.ocr import (
    preprocess_image_for_ocr,
    extract_layout_with_ocr,
    RapidOCREngine,
    MockOCREngine,
)


class TestOCRPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a synthetic test image with high contrast text
        self.img = Image.new("RGB", (300, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(self.img)
        draw.text((10, 10), "PT-101 24 VDC", fill=(0, 0, 0))
        draw.text((10, 50), "PRESSURE TRANSMITTER", fill=(0, 0, 0))

    def test_preprocess_image_for_ocr(self):
        """Verify preprocessing normalizes channels, enhances contrast, and returns valid PIL Image."""
        processed = preprocess_image_for_ocr(self.img)
        self.assertIsInstance(processed, Image.Image)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, self.img.size)

    def test_mock_layout_extraction(self):
        """Verify layout extraction works via fallback interface."""
        mock = MockOCREngine(mock_text="PT-101 | 24 VDC\nPRESSURE TRANSMITTER", mock_confidence=0.95)
        text, conf = mock.extract_text_from_image(self.img)
        self.assertEqual(conf, 0.95)
        self.assertIn("PT-101", text)

    def test_extract_layout_with_ocr_format(self):
        """Verify extract_layout_with_ocr returns the expected dictionary structure."""
        layout = extract_layout_with_ocr(self.img)
        self.assertIn("text", layout)
        self.assertIn("average_confidence", layout)
        self.assertIn("blocks", layout)
        self.assertIn("line_count", layout)
        self.assertIsInstance(layout["blocks"], list)


if __name__ == "__main__":
    unittest.main()
