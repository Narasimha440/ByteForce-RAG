import unittest

from app.rag import verify_numerical_grounding


class TestCRAGGuard(unittest.TestCase):

    def test_numerical_grounding_success(self):
        query = "What is the maximum operating pressure of the HCU reactor loop?"
        mock_hits = [
            {"text": "Reactor Loop Operating Pressure is 165 - 180 bar(g) with high alarm at 188 bar(g)."}
        ]
        result = verify_numerical_grounding(query, mock_hits)
        self.assertTrue(result["grounded"])
        self.assertGreaterEqual(result["measurements_found"], 1)

    def test_numerical_grounding_failure_when_no_numbers(self):
        query = "What is the temperature limit for the crude heater?"
        mock_hits = [
            {"text": "The crude heater should be operated with caution and checked regularly by operators."}
        ]
        result = verify_numerical_grounding(query, mock_hits)
        self.assertFalse(result["grounded"])
        self.assertEqual(result["measurements_found"], 0)

    def test_qualitative_query_skips_guard(self):
        query = "Who is responsible for the turnaround shutdown?"
        mock_hits = [
            {"text": "The shift superintendent coordinates the turnaround shutdown protocol."}
        ]
        result = verify_numerical_grounding(query, mock_hits)
        self.assertTrue(result["grounded"])
        self.assertEqual(result["details"], "General qualitative query.")


if __name__ == "__main__":
    unittest.main()
