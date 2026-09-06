import unittest
from unittest.mock import MagicMock, patch

from app.rag_tools import (
    AGENT_TOOL_SCHEMAS,
    get_airgap_status,
    inspect_scanned_document,
    lookup_equipment_tag,
    search_internal_knowledge,
)


class TestRAGTools(unittest.TestCase):

    def test_agent_tool_schemas_validity(self):
        self.assertIsInstance(AGENT_TOOL_SCHEMAS, list)
        self.assertGreaterEqual(len(AGENT_TOOL_SCHEMAS), 4)

        names = [tool["function"]["name"] for tool in AGENT_TOOL_SCHEMAS]
        self.assertIn("search_internal_knowledge", names)
        self.assertIn("lookup_equipment_tag", names)
        self.assertIn("inspect_scanned_document", names)
        self.assertIn("get_airgap_status", names)

        for tool in AGENT_TOOL_SCHEMAS:
            self.assertEqual(tool["type"], "function")
            self.assertIn("name", tool["function"])
            self.assertIn("description", tool["function"])
            self.assertIn("parameters", tool["function"])

    def test_airgap_status_tool(self):
        status = get_airgap_status()
        self.assertIn("compliant", status)
        self.assertIn("verdict", status)
        self.assertTrue(status["compliant"])

    @patch("app.rag_tools.get_rag")
    def test_search_internal_knowledge_tool(self, mock_get_rag):
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [
            {"filename": "MRPL_HCU.docx", "page_number": 1, "text": "HCU operating limit 170 bar", "tags": ["HCU", "BDV-701"]}
        ]
        mock_rag.build_context.return_value = (
            "[EVIDENCE 1] (Document: MRPL_HCU.docx, Page: 1)\nHCU operating limit 170 bar",
            [{"filename": "MRPL_HCU.docx", "location": "p.1"}]
        )
        mock_get_rag.return_value = mock_rag

        result = search_internal_knowledge("HCU pressure limit")
        self.assertIn("[EVIDENCE 1]", result)
        self.assertIn("MRPL_HCU.docx", result)

    @patch("app.rag_tools.get_rag")
    def test_lookup_equipment_tag_tool(self, mock_get_rag):
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [
            {
                "filename": "MRPL_HCU.docx",
                "page_number": 2,
                "document_type": "01_Standards_Reference",
                "score": 0.85,
                "text": "Emergency depressurization valve BDV-701 must depressurize within 15 mins.",
                "tags": ["BDV-701", "HCU"],
            }
        ]
        mock_get_rag.return_value = mock_rag

        res = lookup_equipment_tag("BDV-701")
        self.assertEqual(res["tag"], "BDV-701")
        self.assertEqual(res["matches_found"], 1)
        self.assertEqual(res["references"][0]["document"], "MRPL_HCU.docx")

    def test_inspect_scanned_document_missing_file(self):
        res = inspect_scanned_document("non_existent_file.pdf")
        self.assertFalse(res["success"])
        self.assertIn("File not found", res["error"])


if __name__ == "__main__":
    unittest.main()
