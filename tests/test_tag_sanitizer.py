import unittest
from app.ocr import sanitize_industrial_tags, extract_text_with_ocr

class TestTagSanitizer(unittest.TestCase):
    def test_sanitize_isa_tags_and_optical_errors(self):
        # O to 0 in numeric field
        self.assertEqual(sanitize_industrial_tags("Transmitter PT-1O1"), "Transmitter PT-101")
        # l/I to 1 in numeric field
        self.assertEqual(sanitize_industrial_tags("Indicator TI-20lA"), "Indicator TI-201A")
        # Valve tag
        self.assertEqual(sanitize_industrial_tags("Emergency valve XV-3Ol"), "Emergency valve XV-301")

    def test_sanitize_electrical_and_process_units(self):
        self.assertEqual(sanitize_industrial_tags("Power supply: 24 V0C"), "Power supply: 24 VDC")
        self.assertEqual(sanitize_industrial_tags("Mains: 12O VAC"), "Mains: 120 VAC")
        self.assertEqual(sanitize_industrial_tags("Signal: 4-2O mA loop"), "Signal: 4-20 mA loop")
        self.assertEqual(sanitize_industrial_tags("Discharge: 3.2 bar (g)"), "Discharge: 3.2 bar(g)")

    def test_preserve_valid_engineering_strings(self):
        original = "Refinery unit operating at 530°C with FT-204 and OISD-156 standards."
        self.assertEqual(sanitize_industrial_tags(original), original)

    def test_extract_text_with_ocr_sanitization(self):
        # Test mock engine with simulated optical error
        from app.ocr import MockOCREngine
        mock = MockOCREngine(mock_text="Sensor PT-1O1 trip limit at 24 V0C")
        text, conf = mock.extract_text_from_image("dummy")
        cleaned = sanitize_industrial_tags(text)
        self.assertIn("PT-101", cleaned)
        self.assertIn("24 VDC", cleaned)

if __name__ == "__main__":
    unittest.main()
