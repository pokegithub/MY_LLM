import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "cli_hardware_validate_contracts_v1.json"
INVENTORY_PATH = REPO_ROOT / "tests" / "fixtures" / "cli_command_inventory_v1.json"
DEFAULT_HARDWARE_REPORT = REPO_ROOT / "run_artifacts" / "hardware_readiness_report.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def assert_no_truthfulness_overclaim(testcase, text):
    lowered = text.lower()
    for forbidden in {
        "phase b started: true",
        "phase_b_started\": true",
        "training started: true",
        "training_started\": true",
        "model quality proven: true",
        "model_quality_claim\": \"proven",
        "training ready: true",
        "ready_for_training\": true",
    }:
        testcase.assertNotIn(forbidden, lowered)


def _default_report_stat():
    if not DEFAULT_HARDWARE_REPORT.exists():
        return None
    stat = DEFAULT_HARDWARE_REPORT.stat()
    return (stat.st_mtime_ns, stat.st_size)


class CLIHardwareValidateContractTests(unittest.TestCase):
    def run_command(self, *args, timeout=120):
        return subprocess.run(
            [sys.executable, "run.py", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_hardware_validate_json_contract_uses_temp_report_path(self):
        fixture = load_fixture()
        before_default = _default_report_stat()
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "hardware_readiness_contract.json"
            result = self.run_command(
                "--json",
                "--report-path",
                str(report_path),
                "hardware-validate",
                timeout=180,
            )

            self.assertTrue(result.stdout.strip(), result.stderr)
            payload = json.loads(result.stdout)
            expected_rc = 0 if payload.get("ok") else 1
            self.assertEqual(result.returncode, expected_rc, result.stderr + result.stdout)
            self.assertTrue(report_path.is_file())
            written = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], fixture["hardware_validate"]["expected_schema"])
        for key in fixture["hardware_validate"]["required_json_keys"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["quality_claim"], "none")
        self.assertEqual(payload["training_readiness_claim"], "none")
        self.assertFalse(payload["hardware_ready_for_training"])
        self.assertEqual(Path(payload["report_path"]).resolve(), report_path.resolve())
        self.assertEqual(written["schema"], payload["schema"])
        self.assertEqual(_default_report_stat(), before_default)
        assert_no_truthfulness_overclaim(self, result.stdout + result.stderr)

    def test_hardware_validate_text_contract_uses_temp_report_path(self):
        fixture = load_fixture()
        before_default = _default_report_stat()
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "hardware_readiness_text_contract.json"
            result = self.run_command(
                "--report-path",
                str(report_path),
                "hardware-validate",
                timeout=180,
            )

            self.assertIn(result.returncode, {0, 1}, result.stderr + result.stdout)
            self.assertTrue(report_path.is_file())

        for marker in fixture["hardware_validate"]["required_text_markers"]:
            self.assertIn(marker, result.stdout)
        self.assertIn(report_path.name, result.stdout)
        self.assertEqual(_default_report_stat(), before_default)
        assert_no_truthfulness_overclaim(self, result.stdout + result.stderr)

    def test_gpu_fit_validate_safe_mocked_json_contract(self):
        fixture = load_fixture()
        self.assertFalse(fixture["gpu_fit_validate"]["real_cuda_execution_allowed_in_unit_tests"])
        self.assertFalse(fixture["gpu_fit_validate"]["direct_subprocess_execution_allowed"])

        import run

        fake_report = _fake_gpu_fit_report()
        fake_train = types.SimpleNamespace(
            run_gpu_constrained_fit_validation=lambda: dict(fake_report)
        )
        old_train = sys.modules.get("train")
        old_output_json = run.OUTPUT_JSON
        old_report_path = run.REPORT_PATH
        try:
            sys.modules["train"] = fake_train
            run.OUTPUT_JSON = True
            with tempfile.TemporaryDirectory() as td:
                report_path = Path(td) / "gpu_fit_contract.json"
                run.REPORT_PATH = str(report_path)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    run.run_gpu_fit_validate()
                payload = json.loads(stdout.getvalue())
                written = json.loads(report_path.read_text(encoding="utf-8"))
        finally:
            run.OUTPUT_JSON = old_output_json
            run.REPORT_PATH = old_report_path
            if old_train is None:
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = old_train

        self.assertEqual(payload["schema"], fixture["gpu_fit_validate"]["expected_schema"])
        for key in fixture["gpu_fit_validate"]["required_json_keys"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["quality_claim"], "none")
        self.assertEqual(payload["pretraining_readiness_changed"], "no")
        self.assertEqual(Path(payload["report_path"]).resolve(), report_path.resolve())
        self.assertEqual(written["schema"], payload["schema"])
        self.assertEqual(payload["matrix"], [])
        assert_no_truthfulness_overclaim(self, json.dumps(payload))

    def test_gpu_fit_validate_safe_mocked_text_contract(self):
        fixture = load_fixture()

        import run

        fake_train = types.SimpleNamespace(
            run_gpu_constrained_fit_validation=_fake_gpu_fit_report
        )
        old_train = sys.modules.get("train")
        old_output_json = run.OUTPUT_JSON
        old_report_path = run.REPORT_PATH
        try:
            sys.modules["train"] = fake_train
            run.OUTPUT_JSON = False
            with tempfile.TemporaryDirectory() as td:
                report_path = Path(td) / "gpu_fit_text_contract.json"
                run.REPORT_PATH = str(report_path)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    run.run_gpu_fit_validate()
                output = stdout.getvalue()
                report_exists = report_path.is_file()
        finally:
            run.OUTPUT_JSON = old_output_json
            run.REPORT_PATH = old_report_path
            if old_train is None:
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = old_train

        for marker in fixture["gpu_fit_validate"]["required_text_markers"]:
            self.assertIn(marker, output)
        self.assertIn(report_path.name, output)
        self.assertTrue(report_exists)
        assert_no_truthfulness_overclaim(self, output)

    def test_gpu_fit_validate_direct_execution_remains_disallowed_by_contract(self):
        fixture = load_fixture()
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

        self.assertIn("gpu-fit-validate", inventory["commands"])
        self.assertEqual(inventory["command_count"], 51)
        self.assertTrue(fixture["gpu_fit_validate"]["safe_execution_required_before_direct_cli_run"])
        self.assertFalse(fixture["gpu_fit_validate"]["real_cuda_execution_allowed_in_unit_tests"])
        self.assertFalse(fixture["gpu_fit_validate"]["direct_subprocess_execution_allowed"])
        self.assertEqual(
            fixture["gpu_fit_validate"]["execution_mode_under_contract"],
            "mocked_handler_only",
        )


def _fake_gpu_fit_report():
    return {
        "schema": "gpu_constrained_fit_validation_v1",
        "ok": True,
        "interpreter": {
            "executable": sys.executable,
            "is_repo_local_venv": True,
        },
        "torch_version": "fake-test",
        "torch_cuda_runtime": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "nvidia_smi": {"gpus": []},
        "gpu_name": None,
        "detected_total_vram_gb": None,
        "vram_class": "unknown_or_larger",
        "preferred_dtype": "unverified_cuda_unavailable",
        "hardware_classification": "validation_only_for_current_repo_target",
        "matrix": [],
        "smallest_fitting_config": None,
        "strongest_fitting_config": None,
        "failed_configs": [],
        "short_validation": None,
        "default_training_fit": {
            "fit_classification": "partially_unverified",
            "preflight_ready_for_training": False,
            "evidence": "fake-test",
        },
        "pretraining_readiness_changed": "no",
        "quality_claim": "none",
        "limitations": [
            "mocked handler contract only; real CUDA fit was not executed"
        ],
        "status": "mocked_contract",
    }


if __name__ == "__main__":
    unittest.main()
