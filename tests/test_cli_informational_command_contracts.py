import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "cli_informational_command_contracts_v1.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def assert_no_truthfulness_overclaim(testcase, text):
    lowered = text.lower()
    for forbidden in {
        "phase b started: true",
        "phase_b_started\": true",
        "training started: true",
        "training_started\": true",
        "sft started: true",
        "dpo started: true",
        "rlvr started: true",
        "model quality proven: true",
        "model_quality_claim\": \"proven",
    }:
        testcase.assertNotIn(forbidden, lowered)


class CLIInformationalCommandContractTests(unittest.TestCase):
    def run_command(self, *args):
        return subprocess.run(
            [sys.executable, "run.py", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def test_deployment_info_json_contract(self):
        fixture = load_fixture()
        result = self.run_command("--json", "deployment-info")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)

        for key in fixture["required_json_keys"]["deployment-info"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["schema"], fixture["truthfulness_expectations"]["deployment_schema"])
        self.assertEqual(payload["quality_claim"], fixture["truthfulness_expectations"]["quality_claim"])
        self.assertIsInstance(payload["reports"], list)
        self.assertGreaterEqual(len(payload["reports"]), 3)

        tier_names = {report["tier"]["name"] for report in payload["reports"]}
        self.assertIn("2gb_edge", tier_names)
        self.assertIn("8gb_laptop", tier_names)
        self.assertIn("16gb_workstation", tier_names)

        for report in payload["reports"]:
            for key in fixture["required_report_keys"]["deployment-info"]:
                self.assertIn(key, report)
            self.assertFalse(report["profile_is_validated_runtime_support"])
            self.assertEqual(report["repo_verified_external_runtimes"], [])
            self.assertEqual(report["estimates"]["estimate_scope"], "lower_bound_only")
            self.assertIn("GGUF/llama.cpp", report["external_unverified_targets"])

        assert_no_truthfulness_overclaim(self, result.stdout)

    def test_deployment_info_text_markers(self):
        result = self.run_command("deployment-info")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for marker in load_fixture()["required_text_markers"]["deployment-info"]:
            self.assertIn(marker, result.stdout)
        assert_no_truthfulness_overclaim(self, result.stdout)

    def test_hw_profile_json_contract(self):
        fixture = load_fixture()
        self.assertFalse(fixture["truthfulness_expectations"]["hw_profile_json_mode_currently_text_only"])
        result = self.run_command("--json", "hw-profile")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)

        for key in fixture["required_json_keys"]["hw-profile"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["schema"], fixture["truthfulness_expectations"]["hardware_profile_schema"])
        self.assertEqual(payload["command"], "hw-profile")
        self.assertIsInstance(payload["torch_available"], bool)
        self.assertIsInstance(payload["devices"], list)
        self.assertIsInstance(payload["warnings"], list)
        self.assertFalse(payload["training_started"])
        self.assertFalse(payload["phase_b_started"])
        self.assertFalse(payload["cloud_used"])
        self.assertEqual(payload["model_quality_claim"], "none")
        self.assertEqual(payload["quality_claim"], "none")
        self.assertNotIn("STEP 0: Hardware Profile", result.stdout)
        assert_no_truthfulness_overclaim(self, result.stdout + result.stderr)

    def test_hw_profile_text_markers(self):
        result = self.run_command("hw-profile")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for marker in load_fixture()["required_text_markers"]["hw-profile"]:
            self.assertIn(marker, result.stdout)
        assert_no_truthfulness_overclaim(self, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
