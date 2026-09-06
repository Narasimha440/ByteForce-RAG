import unittest

from app.equipment_graph import format_dossier_box, get_equipment_dossier


class TestEquipmentGraph(unittest.TestCase):

    def test_bdv_701_dossier(self):
        dossier = get_equipment_dossier("BDV-701")
        self.assertIsNotNone(dossier)
        self.assertEqual(dossier["tag"], "BDV-701")
        self.assertIn("Hydrocracker Unit", dossier["unit"])
        self.assertIn("180 bar", dossier["normal_operating_range"])
        self.assertIn("188 bar", dossier["trip_limit"])
        self.assertIn("MRPL_HCU_Hydrocracker_Operating_SOP.docx", dossier["referenced_documents"])

    def test_xv_301_dossier(self):
        dossier = get_equipment_dossier("XV-301")
        self.assertIsNotNone(dossier)
        self.assertEqual(dossier["tag"], "XV-301")
        self.assertIn("SIL-3", dossier["trip_limit"])
        self.assertIn("5.8 seconds", dossier["statutory_audit_finding"])

    def test_format_dossier_box(self):
        dossier = get_equipment_dossier("BDV-701")
        box = format_dossier_box(dossier)
        self.assertIn("INDUSTRIAL ASSET DOSSIER: BDV-701", box)
        self.assertIn("Normal Operating: 165", box)
        self.assertIn("Cross-Referenced Documents:", box)

    def test_unknown_tag_dynamic_mining(self):
        mock_hits = [
            {
                "filename": "MRPL_Test_SOP.docx",
                "text": "Temperature sensor TT-999 monitors reboiler inlet at 210°C.",
                "tags": ["TT-999"],
            }
        ]
        dossier = get_equipment_dossier("TT-999", rag_hits=mock_hits)
        self.assertIsNotNone(dossier)
        self.assertEqual(dossier["tag"], "TT-999")
        self.assertIn("MRPL_Test_SOP.docx", dossier["referenced_documents"])


if __name__ == "__main__":
    unittest.main()
