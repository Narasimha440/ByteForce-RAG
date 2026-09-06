import unittest
from app.rag import get_visual_evidence, LocalRAG

class TestMultimodalHandoff(unittest.TestCase):
    def test_get_visual_evidence_from_cad_hits(self):
        hits = [
            {
                "filename": "Appendix D-2, C6642-10 Sample P&IDs, 07-Typical Chlorinators.dwg",
                "source": "data/03_PIDs/Appendix D-2, C6642-10 Sample P&IDs, 07-Typical Chlorinators.dwg",
                "page_number": 1,
                "file_type": "dwg",
                "content_type": "cad_drawing",
                "requires_vision_agent": True,
                "drawing_code": "C6642-10",
                "tags": ["PT-101", "XV-301"],
                "ocr_used": False,
                "score": 0.88,
            },
            {
                "filename": "MRPL_CDU_VDU_Crude_Distillation_SOP.docx",
                "source": "data/01_Standards_Reference/MRPL_CDU_VDU_Crude_Distillation_SOP.docx",
                "page_number": 1,
                "file_type": "docx",
                "content_type": "document",
                "requires_vision_agent": False,
                "tags": [],
                "ocr_used": False,
                "score": 0.72,
            }
        ]

        visuals = get_visual_evidence(hits)
        self.assertEqual(len(visuals), 1)
        self.assertEqual(visuals[0]["drawing_code"], "C6642-10")
        self.assertEqual(visuals[0]["file_type"], "dwg")
        self.assertTrue(visuals[0]["requires_vision_agent"])
        self.assertEqual(visuals[0]["target_model"], "Qwen2.5-VL-7B-Instruct")
        self.assertIn("PT-101", visuals[0]["equipment_tags"])

    def test_build_context_enriches_visual_handoff(self):
        hits = [
            {
                "filename": "drawing.dwg",
                "source": "data/03_PIDs/drawing.dwg",
                "page_number": 1,
                "file_type": "dwg",
                "content_type": "cad_drawing",
                "requires_vision_agent": True,
                "drawing_code": "C6642-1",
                "tags": ["XV-301"],
                "ocr_used": False,
                "score": 0.9,
                "text": "Sample CAD Text"
            }
        ]
        rag = LocalRAG()
        context, sources = rag.build_context(hits)
        self.assertEqual(len(sources), 1)
        self.assertIn("visual_handoff", sources[0])
        self.assertTrue(sources[0]["visual_handoff"]["requires_vision_agent"])
        self.assertEqual(sources[0]["visual_handoff"]["target_model"], "Qwen2.5-VL-7B-Instruct")

if __name__ == "__main__":
    unittest.main()
