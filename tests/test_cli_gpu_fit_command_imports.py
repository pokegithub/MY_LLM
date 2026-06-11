import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CLIGPUFitCommandImportTests(unittest.TestCase):
    def test_gpu_fit_command_module_import_is_quiet_and_lazy(self):
        script = textwrap.dedent(
            """
            import contextlib
            import importlib
            import io
            import json
            import sys
            from pathlib import Path

            paths = [
                Path("run_artifacts/gpu_constrained_fit_report.json"),
                Path("run_artifacts/gpu_constrained_fit"),
            ]

            def snapshot(path):
                if not path.exists():
                    return None
                stat = path.stat()
                if path.is_file():
                    return ["file", stat.st_mtime_ns, stat.st_size]
                entries = []
                for item in sorted(path.rglob("*")):
                    item_stat = item.stat()
                    entries.append([
                        str(item.relative_to(path)),
                        "dir" if item.is_dir() else "file",
                        item_stat.st_mtime_ns,
                        item_stat.st_size,
                    ])
                return ["dir", stat.st_mtime_ns, stat.st_size, entries]

            before = {str(path): snapshot(path) for path in paths}
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                module = importlib.import_module("cli.commands.gpu_fit")
            after = {str(path): snapshot(path) for path in paths}

            payload = {
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "train_imported": "train" in sys.modules,
                "torch_imported": "torch" in sys.modules,
                "files_changed": before != after,
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
        self.assertFalse(payload["files_changed"])
        self.assertTrue(payload["has_run_gpu_fit_validate"])


if __name__ == "__main__":
    unittest.main()
