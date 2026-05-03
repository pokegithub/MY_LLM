import json
import os
import unittest


class BackendIntegrationPlanAssetTests(unittest.TestCase):
    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_backend_integration_artifacts_exist(self):
        self.assertTrue(os.path.isfile("./backend_integration/backend_integration_plan_v1.md"))
        self.assertTrue(os.path.isfile("./backend_integration/backend_integration_plan_v1.json"))
        self.assertTrue(os.path.isfile("./backend_integration/backend_capability_gate_v1.md"))
        self.assertTrue(os.path.isfile("./backend_integration/backend_capability_gate_v1.json"))
        self.assertTrue(os.path.isfile("./backend_integration/runtime_assumptions_v1.md"))
        self.assertTrue(os.path.isfile("./backend_integration/backend_failure_contract_v1.md"))
        self.assertTrue(os.path.isfile("./backend_integration/backend_failure_contract_v1.json"))

    def test_exactly_one_primary_backend_path_is_defined(self):
        payload = self._load_json("./backend_integration/backend_integration_plan_v1.json")
        primary = payload["primary_backend_path"]

        self.assertEqual(payload["schema"], "backend_integration_plan_v1")
        self.assertEqual(primary["path_count"], 1)
        self.assertEqual(primary["backend_path_id"], "local_transformers_in_process")
        self.assertEqual(primary["target_environment"], "linux_first_48gb_workstation_class")
        self.assertEqual(primary["status"], "planned_not_integrated")

    def test_capability_gate_has_required_backend_checks(self):
        payload = self._load_json("./backend_integration/backend_capability_gate_v1.json")
        gate_ids = {item["gate_id"] for item in payload["mandatory_gates"]}
        self.assertEqual(payload["schema"], "backend_capability_gate_v1")
        self.assertIn("structured_candidate_output_validity", gate_ids)
        self.assertIn("schema_adherence_parse_reliability", gate_ids)
        self.assertIn("coding_usefulness_beyond_no_backend_baseline", gate_ids)
        self.assertIn("deterministic_exact_task_bypass_preserved", gate_ids)
        self.assertIn("verifier_compatibility", gate_ids)
        self.assertIn("retry_loop_compatibility", gate_ids)
        self.assertIn("runtime_stability_linux_first_workstation", gate_ids)

    def test_failure_contract_has_required_failure_classes(self):
        payload = self._load_json("./backend_integration/backend_failure_contract_v1.json")
        failure_ids = {item["failure_class"] for item in payload["failure_classes"]}
        expected = {
            "backend_not_configured",
            "backend_unavailable",
            "model_load_failed",
            "malformed_candidate_output",
            "schema_validation_failed",
            "runtime_execution_failed",
            "timeout",
            "unsupported_capability",
            "degraded_mode_only",
            "verification_failed_after_candidate",
        }
        self.assertEqual(payload["schema"], "backend_failure_contract_v1")
        self.assertEqual(failure_ids, expected)

    def test_artifacts_preserve_fail_closed_and_exact_bypass_language(self):
        with open("./backend_integration/backend_integration_plan_v1.md", "r", encoding="utf-8") as handle:
            plan_text = handle.read().lower()
        with open("./backend_integration/backend_capability_gate_v1.md", "r", encoding="utf-8") as handle:
            gate_text = handle.read().lower()
        with open("./backend_integration/backend_failure_contract_v1.md", "r", encoding="utf-8") as handle:
            failure_text = handle.read().lower()
        with open("./backend_integration/runtime_assumptions_v1.md", "r", encoding="utf-8") as handle:
            runtime_text = handle.read().lower()

        combined = "\n".join([plan_text, gate_text, failure_text, runtime_text])
        self.assertIn("linux-first", combined)
        self.assertIn("deterministic exact-task bypass", combined)
        self.assertIn("verifier authority", combined)
        self.assertIn("fail-closed", combined)
        self.assertNotIn("already integrated", combined)
        self.assertNotIn("production ready", combined)
        self.assertNotIn("backend may override exact", combined)


if __name__ == "__main__":
    unittest.main()
