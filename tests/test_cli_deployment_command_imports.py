import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CLIDeploymentCommandImportTests(unittest.TestCase):
    def test_deployment_command_module_import_is_quiet_and_lazy(self):
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
                module = importlib.import_module("cli.commands.deployment")

            payload = {
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "torch_imported": "torch" in sys.modules,
                "hardware_profiles_imported": "hardware_profiles" in sys.modules,
                "exports": {
                    name: hasattr(module, name)
                    for name in [
                        "build_deployment_info_report",
                        "run_deployment_info",
                        "build_hw_profile_report",
                        "run_hw_profile",
                    ]
                },
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
        self.assertFalse(payload["torch_imported"])
        self.assertFalse(payload["hardware_profiles_imported"])
        self.assertTrue(all(payload["exports"].values()))


if __name__ == "__main__":
    unittest.main()
