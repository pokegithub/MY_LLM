import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "cloud" / "lightning_ai" / "cloud_readiness_runner.py"
POLICY_PATH = REPO_ROOT / "cloud" / "lightning_ai" / "gpu_policy_v1.json"
CONFIG_PATH = REPO_ROOT / "configs" / "cloud_candidate_readiness_example.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("lightning_cloud_readiness_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LightningCloudPrepTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()
        self.policy = self.runner.load_policy(POLICY_PATH)

    def make_args(self, *argv):
        return self.runner.build_arg_parser().parse_args(list(argv))

    def build_report(self, args, env=None):
        return self.runner.evaluate_guards(
            policy=self.policy,
            args=args,
            env=env or {},
            include_torch=False,
        )

    def test_default_mode_is_dry_run_and_no_cloud_job_starts(self):
        args = self.make_args()
        report = self.build_report(args)
        self.assertEqual(args.mode, "dry_run")
        self.assertFalse(report["execute_requested"])
        self.assertFalse(report["execution_allowed"])
        self.assertFalse(report["cloud_job_started"])
        self.assertEqual(report["credits_spent"], 0)

    def test_execute_requires_explicit_confirmation_and_auth(self):
        args = self.make_args("--run-smoke", "--execute")
        report = self.build_report(args, env={})
        self.assertTrue(report["execute_requested"])
        self.assertFalse(report["execution_allowed"])
        self.assertIn("manual_execute_confirmation_missing", report["blocked_reasons"])
        self.assertIn("auth_missing", report["blocked_reasons"])
        self.assertIn("free_credit_balance_unknown", report["blocked_reasons"])

    def test_unknown_credit_balance_fails_closed_for_execute(self):
        args = self.make_args("--run-smoke", "--execute")
        env = {
            "MYLLM_CLOUD_CONFIRM_EXECUTE": "YES",
            "MYLLM_CLOUD_MAX_CREDITS": "15",
            "MYLLM_LIGHTNING_AUTH_CONFIRMED": "YES",
            "MYLLM_LIGHTNING_PRICE_CONFIRMED": "YES",
        }
        report = self.build_report(args, env=env)
        self.assertFalse(report["execution_allowed"])
        self.assertIn("free_credit_balance_unknown", report["blocked_reasons"])

    def test_paid_overage_is_disallowed_and_budget_is_enforced(self):
        self.assertFalse(self.policy["allow_paid_overage"])
        self.assertTrue(self.policy["require_free_credit_only"])
        self.assertEqual(self.policy["credit_budget_max"], 15)

        args = self.make_args("--run-smoke", "--execute")
        env = {
            "MYLLM_CLOUD_CONFIRM_EXECUTE": "YES",
            "MYLLM_CLOUD_MAX_CREDITS": "15",
            "MYLLM_LIGHTNING_AUTH_CONFIRMED": "YES",
            "MYLLM_LIGHTNING_FREE_CREDITS_AVAILABLE": "15",
            "MYLLM_LIGHTNING_ESTIMATED_CREDITS": "16",
        }
        report = self.build_report(args, env=env)
        self.assertFalse(report["execution_allowed"])
        self.assertIn("estimated_cost_exceeds_budget", report["blocked_reasons"])

    def test_training_task_classes_are_blocked(self):
        for task_class in self.policy["disallowed_task_classes"]:
            args = self.make_args("--task-class", task_class)
            report = self.build_report(args)
            self.assertFalse(report["task"]["allowed"])
            self.assertTrue(report["task"]["disallowed"])
            self.assertIn("task_class_not_allowed", report["blocked_reasons"])

    def test_allowed_task_classes_are_allowed_in_dry_run(self):
        for task_class in self.policy["allowed_task_classes"]:
            args = self.make_args("--task-class", task_class)
            report = self.build_report(args)
            self.assertTrue(report["task"]["allowed"])
            self.assertFalse(report["task"]["disallowed"])

    def test_plan_contains_no_training_commands(self):
        args = self.make_args("--print-plan")
        report = self.build_report(args)
        commands = "\n".join(report["plan"]["commands"]).lower()
        for blocked in (" run.py train", " run.py sft", " run.py dpo", " run.py distill", " run.py improve"):
            self.assertNotIn(blocked, commands)
        self.assertIn("python run.py candidate-readiness-smoke", report["plan"]["commands"])
        self.assertTrue(report["plan"]["training_commands_blocked"])

    def test_policy_and_config_do_not_contain_secret_values(self):
        for path in (POLICY_PATH, CONFIG_PATH):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("password=", text)
            self.assertNotIn("token=", text)
            self.assertNotIn("cookie=", text)
            self.assertNotIn("sk-", text)

    def test_execute_can_pass_guards_only_with_free_credit_confirmation(self):
        args = self.make_args("--run-smoke", "--execute")
        env = {
            "MYLLM_CLOUD_CONFIRM_EXECUTE": "YES",
            "MYLLM_CLOUD_MAX_CREDITS": "15",
            "MYLLM_LIGHTNING_AUTH_CONFIRMED": "YES",
            "MYLLM_LIGHTNING_FREE_CREDITS_AVAILABLE": "15",
            "MYLLM_LIGHTNING_ESTIMATED_CREDITS": "2",
        }
        report = self.build_report(args, env=env)
        self.assertTrue(report["execution_allowed"])
        self.assertEqual(report["budget"]["credit_budget_max"], 15.0)
        self.assertFalse(report["budget"]["allow_paid_overage"])


if __name__ == "__main__":
    unittest.main()
