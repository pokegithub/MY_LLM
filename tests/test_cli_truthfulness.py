import subprocess
import sys
import unittest
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CLITruthfulnessTests(unittest.TestCase):
    def run_command(self, *args):
        return subprocess.run(
            [sys.executable, "run.py", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_status_runs_with_ascii_safe_output(self):
        result = self.run_command("status")
        self.assertEqual(result.returncode, 0, result.stderr)
        combined = result.stdout + result.stderr
        self.assertIn("PIPELINE STATUS", result.stdout)
        self.assertIn("DEPENDENCY ENVIRONMENT", result.stdout)
        self.assertNotIn("Traceback", combined)
        self.assertNotIn("\u2713", combined)
        self.assertNotIn("\u2717", combined)
        self.assertNotIn("NEXT: python run.py train\n", result.stdout)

    def test_audit_runs_pass1_truthfulness_checks(self):
        result = self.run_command("audit")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Status command health", result.stdout)
        self.assertIn("Fake serving detection", result.stdout)
        self.assertIn("Quantization report integrity", result.stdout)
        self.assertIn("AUDIT PASSED", result.stdout)

    def test_benchmark_harness_help_marks_scale_comparison_unverified(self):
        result = self.run_command("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("external scale comparison disabled/unverified", result.stdout)
        self.assertNotIn("1x/10x/100x comparison report", result.stdout)

    def test_deps_json_reports_machine_readable_environment(self):
        result = self.run_command("deps", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertIn("items", payload)

    def test_status_json_reports_next_action(self):
        result = self.run_command("status", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema"], "pipeline_status_v2")
        self.assertIn("dependency_environment", payload)
        self.assertIn("next_action", payload)

    def test_validate_real_path_json_is_smoke_only(self):
        result = self.run_command("validate-real-path", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["quality_claim"], "none")
        self.assertEqual(payload["data_loader_smoke"]["status"], "pass")
        self.assertIn("report_path", payload)

    def test_token_integrity_json_is_truthful(self):
        result = self.run_command("token-integrity", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["quality_claim"], "none")
        self.assertGreater(payload["artifact_count"], 0)

    def test_agent_exact_task_json_is_deterministic(self):
        result = self.run_command(
            "agent-solve",
            "--task",
            'count substring "ana" in "banana"',
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["route"]["route"], "exact_symbolic")
        self.assertEqual(payload["status"], "verified_success")
        self.assertEqual(payload["quality_claim"], "verification_passed")

    def test_agent_plan_reports_plan_ready_not_blocked(self):
        result = self.run_command(
            "agent-plan",
            "--task",
            "arithmetic: 2 + 2 * 5",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["route"]["route"], "exact_symbolic")
        self.assertEqual(payload["status"], "plan_ready")
        self.assertEqual(payload["quality_claim"], "none")

    def test_agent_coding_task_without_backend_fails_closed(self):
        result = self.run_command(
            "agent-solve",
            "--task",
            "Fix the bug in tests/test_cli_truthfulness.py",
            "--file-hint",
            "tests/test_cli_truthfulness.py",
            "--check",
            "compileall:tests/test_cli_truthfulness.py",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["route"]["route"], "coding_edit")
        self.assertEqual(payload["status"], "blocked_unverified")
        self.assertIn("no coding backend configured", payload["blocked_reason"])


if __name__ == "__main__":
    unittest.main()
