import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CLIHardwareCommandImportTests(unittest.TestCase):
    def test_hardware_command_module_import_is_quiet_and_lazy(self):
        script = textwrap.dedent(
            """
            import contextlib
            import importlib
            import io
            import json
            import sys

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                module = importlib.import_module("cli.commands.hardware")

            payload = {
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "train_imported": "train" in sys.modules,
                "torch_imported": "torch" in sys.modules,
                "has_run_hardware_validate": hasattr(module, "run_hardware_validate"),
                "has_run_gpu_fit_validate": hasattr(module, "run_gpu_fit_validate"),
            }
            print(json.dumps(payload, sort_keys=True))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["stdout"], "")
        self.assertEqual(payload["stderr"], "")
        self.assertFalse(payload["train_imported"])
        self.assertFalse(payload["torch_imported"])
        self.assertTrue(payload["has_run_hardware_validate"])
        self.assertFalse(payload["has_run_gpu_fit_validate"])


if __name__ == "__main__":
    unittest.main()
