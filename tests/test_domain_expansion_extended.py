"""
Unit tests for extended refinery ontology instruments and bidirectional acronym expansion.
"""

import unittest
from app.domain_expansion import (
    REFINERY_ONTOLOGY,
    expand_equipment_acronyms,
    expand_refinery_query,
)


class TestDomainExpansionExtended(unittest.TestCase):

    def test_new_instruments_in_ontology(self):
        """Verify new industrial valves and transmitters are present."""
        expected_instruments = ["PSV", "MOV", "XV", "PT", "TT", "FT", "LT", "HIPPS", "SCADA", "DCS", "SIS"]
        for inst in expected_instruments:
            self.assertIn(inst, REFINERY_ONTOLOGY, f"Missing instrument: {inst}")
            entry = REFINERY_ONTOLOGY[inst]
            self.assertTrue(len(entry["full_name"]) > 0)
            self.assertTrue(len(entry["synonyms"]) > 0)

    def test_expand_equipment_acronyms_spoken_to_code(self):
        """Verify spoken phrases are enriched with exact codes."""
        q1 = "What is the status of the blowdown valve?"
        expanded1 = expand_equipment_acronyms(q1)
        self.assertIn("BDV", expanded1)

        q2 = "Check the emergency isolation valve trip limit"
        expanded2 = expand_equipment_acronyms(q2)
        self.assertIn("XV", expanded2)

        q3 = "Pressure transmitter reading"
        expanded3 = expand_equipment_acronyms(q3)
        self.assertIn("PT", expanded3)

    def test_expand_equipment_acronyms_code_to_full(self):
        """Verify codes are enriched with full engineering names."""
        q = "Check PT-101 and BDV"
        expanded = expand_equipment_acronyms(q)
        self.assertIn("Blowdown Valve", expanded)

    def test_expand_refinery_query_boosts(self):
        """Verify lexical boosts are generated for new instruments."""
        augmented, boosts = expand_refinery_query("PSV set pressure test")
        self.assertIn("Pressure Safety Valve", boosts)


if __name__ == "__main__":
    unittest.main()
