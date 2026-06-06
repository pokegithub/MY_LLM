import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "cli_low_risk_command_contracts_v1.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class CLILowRiskCommandContractTests(unittest.TestCase):
    def run_command(self, *args):
        return subprocess.run(
            [sys.executable, "run.py", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def parse_json_command(self, command):
        result = self.run_command("--json", command)
        if result.returncode != 0:
            self.fail(result.stderr + result.stdout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self.fail(f"{command} did not emit valid JSON: {exc}\n{result.stdout}")

    def test_status_json_contract(self):
        fixture = load_fixture()
        payload = self.parse_json_command("status")
        for key in fixture["required_json_keys"]["status"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["schema"], "pipeline_status_v2")
        self.assertEqual(payload["training_quality_claim"], "none")
        self.assertFalse(payload["next_action"].get("ready_for_training", False))

        status_names = {item["name"] for item in payload["status_checks"]}
        for name in {
            "Tokenizer",
            "Data cache",
            "Pretrain ckpt",
            "SFT ckpt",
            "DPO ckpt",
            "Improved ckpt",
            "Eval results",
        }:
            self.assertIn(name, status_names)

        backend = payload["optional_backend_dependency_environment"]
        self.assertTrue(backend.get("optional"))
        self.assertEqual(backend.get("purpose"), "local_transformers_backend")

    def test_deps_json_contract(self):
        fixture = load_fixture()
        payload = self.parse_json_command("deps")
        for key in fixture["required_json_keys"]["deps"]:
            self.assertIn(key, payload)
        self.assertTrue(payload["exists"])
        self.assertTrue(payload["ok"])
        for key in {"pass", "fail", "unverified"}:
            self.assertIn(key, payload["summary"])
        self.assertIn("optional_backend_dependency_environment", payload)
        self.assertNotIn("model_quality_claim", payload)
        self.assertNotIn("phase_b_started", payload)
        self.assertNotIn("training_started", payload)

    def test_compile_source_json_contract(self):
        fixture = load_fixture()
        payload = self.parse_json_command("compile-source")
        for key in fixture["required_json_keys"]["compile-source"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["schema"], "source_compile_report_v1")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["scope"], "tracked_python_sources_only")
        self.assertFalse(payload["full_compileall_replacement"])
        self.assertFalse(payload["behavior_changed"])
        self.assertIsInstance(payload["failures"], list)
        skipped_roots = set(payload["skipped_generated_roots"])
        for root in {
            ".venv",
            ".git",
            "run_artifacts",
            "data_cache",
            "checkpoints",
            "sft_checkpoints",
            "dpo_checkpoints",
            "distill_checkpoints",
            "quantized",
            "eval_results",
        }:
            self.assertIn(root, skipped_roots)

    def test_status_text_markers(self):
        result = self.run_command("status")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for marker in load_fixture()["required_text_markers"]["status"]:
            self.assertIn(marker, result.stdout)

    def test_deps_text_markers(self):
        result = self.run_command("deps")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for marker in load_fixture()["required_text_markers"]["deps"]:
            self.assertIn(marker, result.stdout)

    def test_compile_source_text_markers(self):
        result = self.run_command("compile-source")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for marker in load_fixture()["required_text_markers"]["compile-source"]:
            self.assertIn(marker, result.stdout)


if __name__ == "__main__":
    unittest.main()
