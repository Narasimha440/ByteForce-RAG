import unittest

from app.domain_expansion import expand_refinery_query, extract_query_equipment_tags


class TestDomainExpansion(unittest.TestCase):

    def test_acronym_expansion_hcu(self):
        query = "What is the HCU operating limit?"
        augmented, boosts = expand_refinery_query(query)
        self.assertIn("Hydrocracker Unit", augmented)
        self.assertIn("Hydrocracker Unit", boosts)
        self.assertTrue(any("quench" in b or "165-180" in augmented for b in boosts + [augmented]))

    def test_acronym_expansion_eds(self):
        query = "EDS protocol during runaway delta-T"
        augmented, boosts = expand_refinery_query(query)
        self.assertIn("Emergency Depressurization System", augmented)
        self.assertTrue(any("BDV-701" in b for b in boosts) or "BDV-701" in augmented)

    def test_fuel_standard_bs6(self):
        query = "What is the sulfur limit in BS6 diesel?"
        augmented, boosts = expand_refinery_query(query)
        self.assertTrue("Bharat Stage VI" in augmented or "BS-VI" in boosts)

    def test_extract_equipment_tags(self):
        query = "Inspect BDV-701 and XV-301 pressure on PT-101 before opening V-102"
        tags = extract_query_equipment_tags(query)
        self.assertIn("BDV-701", tags)
        self.assertIn("XV-301", tags)
        self.assertIn("PT-101", tags)
        self.assertIn("V-102", tags)

    def test_empty_query(self):
        augmented, boosts = expand_refinery_query("")
        self.assertEqual(augmented, "")
        self.assertEqual(boosts, [])


if __name__ == "__main__":
    unittest.main()
