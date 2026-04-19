import unittest
from unittest import mock

import run


class AuditTruthfulnessTests(unittest.TestCase):
    def test_audit_status_command_health_passes(self):
        run.audit_status_command_health()

    def test_audit_serving_contract_rejects_echo_success(self):
        with mock.patch(
            "serving.server.handle_request",
            return_value={"ok": True, "response": "Echo: hello"},
        ):
            with self.assertRaisesRegex(RuntimeError, "echo"):
                run.audit_serving_contract()

    def test_audit_serving_contract_rejects_success_without_backend(self):
        with mock.patch(
            "serving.server.handle_request",
            return_value={"ok": True, "response": "model output"},
        ):
            with self.assertRaisesRegex(RuntimeError, "without a configured model backend"):
                run.audit_serving_contract()

    def test_audit_quantization_report_contract_passes(self):
        run.audit_quantization_report_contract()


if __name__ == "__main__":
    unittest.main()
