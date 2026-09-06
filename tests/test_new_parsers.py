import unittest
from pathlib import Path
from app.parsers import parse_file, parse_pptx, parse_dwg

class TestNewParsers(unittest.TestCase):
    def test_parse_pptx_real_file(self):
        pptx_path = Path("data/02_Templates_Checklists_Forms/Appendix B-9 - Commissioning Flow Chart.pptx")
        if pptx_path.exists():
            records = parse_pptx(pptx_path)
            self.assertGreater(len(records), 0)
            self.assertEqual(records[0]["file_type"], "pptx")
            self.assertEqual(records[0]["content_type"], "presentation")
            self.assertIn("Commissioning Flow Chart", records[0]["text"])

    def test_parse_dwg_real_file(self):
        dwg_path = Path("data/03_PIDs/Appendix D-2, C6642-10 Sample P&IDs, 07-Typical Chlorinators.dwg")
        if dwg_path.exists():
            records = parse_dwg(dwg_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["file_type"], "dwg")
            self.assertEqual(records[0]["content_type"], "cad_drawing")
            self.assertTrue(records[0]["requires_vision_agent"])
            self.assertEqual(records[0]["drawing_code"], "C6642-10")
            self.assertIn("P&ID", records[0]["text"])

    def test_parse_file_dispatch(self):
        pptx_path = Path("data/02_Templates_Checklists_Forms/Appendix B-9 - Commissioning Flow Chart.pptx")
        if pptx_path.exists():
            records = parse_file(pptx_path)
            self.assertGreater(len(records), 0)
            self.assertEqual(records[0]["file_type"], "pptx")

if __name__ == "__main__":
    unittest.main()
