import tempfile
import unittest
from pathlib import Path

from app.airgap import AirGapAuditor


class TestAirGapAuditor(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "test_airgap.log"
        self.auditor = AirGapAuditor(log_path=self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_local_endpoint_detection(self):
        self.assertTrue(AirGapAuditor.is_local_endpoint("127.0.0.1"))
        self.assertTrue(AirGapAuditor.is_local_endpoint("localhost"))
        self.assertTrue(AirGapAuditor.is_local_endpoint("192.168.1.50"))
        self.assertTrue(AirGapAuditor.is_local_endpoint("10.0.0.12"))
        self.assertTrue(AirGapAuditor.is_local_endpoint(""))

        # External endpoints must be flagged as non-local
        self.assertFalse(AirGapAuditor.is_local_endpoint("api.openai.com"))
        self.assertFalse(AirGapAuditor.is_local_endpoint("8.8.8.8"))
        self.assertFalse(AirGapAuditor.is_local_endpoint("huggingface.co"))

    def test_log_local_event(self):
        event = self.auditor.log_event("TEST", "local_search", "127.0.0.1", 6333, "Test local search")
        self.assertEqual(event["status"], "VERIFIED_LOCAL_SOVEREIGN")
        self.assertTrue(event["is_local"])
        self.assertEqual(self.auditor.local_operations, 1)
        self.assertEqual(self.auditor.external_attempts, 0)
        self.assertTrue(self.log_path.exists())

    def test_log_external_event_flags_violation(self):
        event = self.auditor.log_event("TEST", "external_call", "api.openai.com", 443, "Unauthorized")
        self.assertEqual(event["status"], "EXTERNAL_ACCESS_FLAGGED")
        self.assertFalse(event["is_local"])
        self.assertEqual(self.auditor.external_attempts, 1)

        status = self.auditor.get_compliance_status()
        self.assertFalse(status["compliant"])
        self.assertEqual(status["verdict"], "AIR-GAP VIOLATION DETECTED")


if __name__ == "__main__":
    unittest.main()
