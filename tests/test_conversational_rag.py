import unittest
from app.rag import LocalRAG, is_conversational_query, calculate_document_authority, INSUFFICIENT_KNOWLEDGE_MESSAGE


class TestConversationalRAG(unittest.TestCase):

    def test_conversational_intent_detection(self):
        self.assertTrue(is_conversational_query("hello"))
        self.assertTrue(is_conversational_query("Hi there!"))
        self.assertTrue(is_conversational_query("Good morning"))
        self.assertTrue(is_conversational_query("who are you"))
        self.assertTrue(is_conversational_query("what can you do"))
        self.assertTrue(is_conversational_query("thank you"))
        self.assertTrue(is_conversational_query("bye"))

        # Technical queries should NOT be classified as conversational
        self.assertFalse(is_conversational_query("What is the trip limit for PT-101?"))
        self.assertFalse(is_conversational_query("reports"))
        self.assertFalse(is_conversational_query("What were the safety audit findings for XV-301?"))
        self.assertFalse(is_conversational_query("What is the closing time of ESD-XV-001?"))

    def test_document_authority_weighting(self):
        safety_audit_score = calculate_document_authority(
            "MRPL_MAH_Factory_Annual_Safety_Audit_2026.pdf",
            "data/09_Inspection_Accident",
            "reports"
        )
        sop_score = calculate_document_authority(
            "Refinery_SOP_Startup.pdf",
            "data/04_SOPs_Manuals",
            "reports"
        )
        template_score = calculate_document_authority(
            "FAT_SAT_Checklist_Blank_Template.xlsx",
            "data/02_Templates_Checklists_Forms",
            "reports"
        )

        self.assertGreater(safety_audit_score, 1.3)
        self.assertGreater(sop_score, 1.0)
        self.assertLess(template_score, 1.0)
        self.assertGreater(safety_audit_score, template_score)

    def test_conversational_answer_fallback(self):
        from unittest.mock import MagicMock
        mock_embedder = MagicMock()
        mock_embedder.get_dimension.return_value = 1024
        mock_store = MagicMock()
        rag = LocalRAG(embedding_provider=mock_embedder, vector_store=mock_store)
        answer, sources = rag.answer("hello")
        self.assertNotEqual(answer, INSUFFICIENT_KNOWLEDGE_MESSAGE)
        self.assertTrue(any(term in answer for term in ["Sovereign Industrial AI Assistant", "MRPL", "refinery", "operations"]))
        self.assertEqual(len(sources), 0)


if __name__ == "__main__":
    unittest.main()
