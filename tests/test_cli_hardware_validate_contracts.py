import contextlib
import io
import json
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
DEFAULT_GPU_FIT_REPORT = REPO_ROOT / "run_artifacts" / "gpu_constrained_fit_report.json"
GPU_FIT_ARTIFACT_ROOT = REPO_ROOT / "run_artifacts" / "gpu_constrained_fit"


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


def _path_snapshot(path):
    if not path.exists():
        return None
    if path.is_file():
        stat = path.stat()
        return {
            "kind": "file",
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    entries = []
    for item in sorted(path.rglob("*")):
        stat = item.stat()
        entries.append((
            str(item.relative_to(path)),
            "dir" if item.is_dir() else "file",
            stat.st_mtime_ns,
            stat.st_size,
        ))
    stat = path.stat()
    return {
        "kind": "dir",
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "entries": entries,
    }


def _assert_only_expected_temp_files(testcase, temp_dir, expected_names):
    actual = sorted(
        str(path.relative_to(temp_dir))
        for path in temp_dir.rglob("*")
        if path.is_file()
    )
    testcase.assertEqual(actual, sorted(expected_names))


def _emit_json(payload):
    print(json.dumps(payload, sort_keys=True))


def _write_json_report(payload, path):
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(report_path)


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
        before_default = _path_snapshot(DEFAULT_GPU_FIT_REPORT)
        before_artifact_root = _path_snapshot(GPU_FIT_ARTIFACT_ROOT)

        from cli.commands import gpu_fit

        fake_report = _fake_gpu_fit_report()
        fake_train = types.SimpleNamespace(
            run_gpu_constrained_fit_validation=lambda: dict(fake_report)
        )
        old_train = sys.modules.get("train")
        try:
            sys.modules["train"] = fake_train
            with tempfile.TemporaryDirectory() as td:
                temp_dir = Path(td)
                report_path = temp_dir / "gpu_fit_contract.json"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    gpu_fit.run_gpu_fit_validate(
                        True,
                        _emit_json,
                        _write_json_report,
                        str(report_path),
                        str(DEFAULT_GPU_FIT_REPORT),
                    )
                payload = json.loads(stdout.getvalue())
                written = json.loads(report_path.read_text(encoding="utf-8"))
                _assert_only_expected_temp_files(
                    self,
                    temp_dir,
                    [report_path.name],
                )
        finally:
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
        self.assertEqual(_path_snapshot(DEFAULT_GPU_FIT_REPORT), before_default)
        self.assertEqual(_path_snapshot(GPU_FIT_ARTIFACT_ROOT), before_artifact_root)
        assert_no_truthfulness_overclaim(self, json.dumps(payload))

    def test_gpu_fit_validate_safe_mocked_text_contract(self):
        fixture = load_fixture()
        before_default = _path_snapshot(DEFAULT_GPU_FIT_REPORT)
        before_artifact_root = _path_snapshot(GPU_FIT_ARTIFACT_ROOT)

        from cli.commands import gpu_fit

        fake_train = types.SimpleNamespace(
            run_gpu_constrained_fit_validation=_fake_gpu_fit_report
        )
        old_train = sys.modules.get("train")
        try:
            sys.modules["train"] = fake_train
            with tempfile.TemporaryDirectory() as td:
                temp_dir = Path(td)
                report_path = temp_dir / "gpu_fit_text_contract.json"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    gpu_fit.run_gpu_fit_validate(
                        False,
                        _emit_json,
                        _write_json_report,
                        str(report_path),
                        str(DEFAULT_GPU_FIT_REPORT),
                    )
                output = stdout.getvalue()
                report_exists = report_path.is_file()
                _assert_only_expected_temp_files(
                    self,
                    temp_dir,
                    [report_path.name],
                )
        finally:
            if old_train is None:
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = old_train

        for marker in fixture["gpu_fit_validate"]["required_text_markers"]:
            self.assertIn(marker, output)
        self.assertIn(report_path.name, output)
        self.assertTrue(report_exists)
        self.assertEqual(_path_snapshot(DEFAULT_GPU_FIT_REPORT), before_default)
        self.assertEqual(_path_snapshot(GPU_FIT_ARTIFACT_ROOT), before_artifact_root)
        assert_no_truthfulness_overclaim(self, output)

    def test_gpu_fit_validate_mocked_json_contract_exits_nonzero_when_not_ok(self):
        fixture = load_fixture()
        self.assertTrue(fixture["gpu_fit_validate"]["ok_false_exit_behavior_required"])
        before_default = _path_snapshot(DEFAULT_GPU_FIT_REPORT)
        before_artifact_root = _path_snapshot(GPU_FIT_ARTIFACT_ROOT)

        from cli.commands import gpu_fit

        fake_report = _fake_gpu_fit_report(ok=False)
        fake_train = types.SimpleNamespace(
            run_gpu_constrained_fit_validation=lambda: dict(fake_report)
        )
        old_train = sys.modules.get("train")
        try:
            sys.modules["train"] = fake_train
            with tempfile.TemporaryDirectory() as td:
                temp_dir = Path(td)
                report_path = temp_dir / "gpu_fit_not_ok_contract.json"
                stdout = io.StringIO()
                with self.assertRaises(SystemExit) as cm:
                    with contextlib.redirect_stdout(stdout):
                        gpu_fit.run_gpu_fit_validate(
                            True,
                            _emit_json,
                            _write_json_report,
                            str(report_path),
                            str(DEFAULT_GPU_FIT_REPORT),
                        )
                self.assertEqual(cm.exception.code, 1)
                payload = json.loads(stdout.getvalue())
                written = json.loads(report_path.read_text(encoding="utf-8"))
                _assert_only_expected_temp_files(
                    self,
                    temp_dir,
                    [report_path.name],
                )
        finally:
            if old_train is None:
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = old_train

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], fixture["gpu_fit_validate"]["expected_schema"])
        for key in fixture["gpu_fit_validate"]["required_json_keys"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["quality_claim"], "none")
        self.assertEqual(payload["pretraining_readiness_changed"], "no")
        self.assertEqual(Path(payload["report_path"]).resolve(), report_path.resolve())
        self.assertEqual(written["schema"], payload["schema"])
        self.assertFalse(written["ok"])
        self.assertEqual(_path_snapshot(DEFAULT_GPU_FIT_REPORT), before_default)
        self.assertEqual(_path_snapshot(GPU_FIT_ARTIFACT_ROOT), before_artifact_root)
        assert_no_truthfulness_overclaim(self, json.dumps(payload))

    def test_gpu_fit_validate_direct_execution_remains_disallowed_by_contract(self):
        fixture = load_fixture()
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

        self.assertIn("gpu-fit-validate", inventory["commands"])
        self.assertEqual(inventory["command_count"], 51)
        self.assertEqual(fixture["gpu_fit_validate"]["future_module_name"], "cli.commands.gpu_fit")
        self.assertTrue(fixture["gpu_fit_validate"]["safe_execution_required_before_direct_cli_run"])
        self.assertFalse(fixture["gpu_fit_validate"]["real_cuda_execution_allowed_in_unit_tests"])
        self.assertFalse(fixture["gpu_fit_validate"]["direct_subprocess_execution_allowed"])
        self.assertTrue(fixture["gpu_fit_validate"]["temp_report_required"])
        self.assertTrue(fixture["gpu_fit_validate"]["no_default_report_path_write_required"])
        self.assertTrue(fixture["gpu_fit_validate"]["no_checkpoint_artifact_creation_required"])
        self.assertTrue(fixture["gpu_fit_validate"]["future_module_import_safety_required"])
        self.assertTrue(fixture["gpu_fit_validate"]["future_module_must_not_import_train_on_import"])
        self.assertTrue(fixture["gpu_fit_validate"]["future_module_must_not_import_torch_on_import"])
        self.assertTrue(fixture["gpu_fit_validate"]["future_module_must_not_inspect_cuda_on_import"])
        self.assertTrue(fixture["gpu_fit_validate"]["future_module_must_not_write_files_on_import"])
        self.assertEqual(
            fixture["gpu_fit_validate"]["execution_mode_under_contract"],
            "mocked_handler_only",
        )


def _fake_gpu_fit_report(ok=True):
    return {
        "schema": "gpu_constrained_fit_validation_v1",
        "ok": ok,
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
        "status": "mocked_contract" if ok else "mocked_contract_failure",
    }


if __name__ == "__main__":
    unittest.main()
