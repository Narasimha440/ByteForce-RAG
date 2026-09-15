"""
Unit tests for Adaptive RRF weighting, revision authority scoring, and metadata pre-filtering.
"""

import unittest
from app.rag import calculate_document_authority, LocalRAG
from app.domain_expansion import extract_query_equipment_tags


class TestAdaptiveRRF(unittest.TestCase):

    def test_query_tag_detection(self):
        """Verify tag detection for adaptive weighting."""
        tag_queries = [
            "What is the trip limit for PT-101?",
            "What are the inspection records for XV-301?",
            "Check status of BDV-701",
        ]
        conceptual_queries = [
            "What are the general safety guidelines for high pressure?",
            "Describe the overall refinery process flow",
            "What are the operating duties during a shift handover?",
        ]

        for q in tag_queries:
            tags = extract_query_equipment_tags(q)
            self.assertTrue(len(tags) > 0, f"Failed to detect tag in query: {q}")

        for q in conceptual_queries:
            tags = extract_query_equipment_tags(q)
            self.assertEqual(len(tags), 0, f"False positive tag in query: {q}")

    def test_document_authority_with_revisions(self):
        """Verify document authority multipliers factor in revision status."""
        fn = "sop_emergency_shutdown.pdf"
        loc = "04_sops_manuals"
        query = "emergency shutdown"

        # Active / Standard
        auth_active = calculate_document_authority(fn, loc, query, revision="Active")
        # Rev 2.0 (formal revision boost)
        auth_rev = calculate_document_authority(fn, loc, query, revision="Rev 2.0")
        # Draft (preliminary penalty)
        auth_draft = calculate_document_authority(fn, loc, query, revision="Draft")
        # Superseded (obsolete penalty)
        auth_superseded = calculate_document_authority(fn, loc, query, revision="Superseded")

        self.assertGreater(auth_rev, auth_active, "Formally revised document should be boosted")
        self.assertLess(auth_draft, auth_active, "Draft document should have lower authority than active")
        self.assertLess(auth_superseded, auth_draft, "Superseded document should have lowest authority")
        self.assertAlmostEqual(auth_superseded, round(auth_active * 0.40, 4), places=2)

    def test_document_authority_blank_templates(self):
        """Verify blank checklist templates receive significant demotion."""
        fn = "checklist_blank_template.xlsx"
        loc = "02_templates_checklists_forms"
        query = "pump vibration limits"

        auth = calculate_document_authority(fn, loc, query, revision="Active")
        self.assertLess(auth, 1.0, "Blank templates should have authority < 1.0")


if __name__ == "__main__":
    unittest.main()
