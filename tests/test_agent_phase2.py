import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager

from agent.orchestrator import solve_task
from agent.types import (
    SOLVE_STATUS_BLOCKED,
    SOLVE_STATUS_FAILED,
    SOLVE_STATUS_VERIFIED,
    TaskRequest,
)


@contextmanager
def writable_tempdir():
    base_dir = os.path.join(os.getcwd(), "run_artifacts", "test_agent_phase2_workspaces")
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"case_{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_backend(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class AgentPhase2Tests(unittest.TestCase):
    def test_repair_loop_can_turn_failed_initial_candidate_into_verified_success(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
            backend_path = os.path.join(td, "repair_success.json")
            with open(module_path, "w", encoding="utf-8") as handle:
                handle.write("def add(a, b):\n    return a - b\n")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import unittest\n"
                    "from demo import add\n\n"
                    "class DemoTests(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(2, 3), 5)\n"
                )
            write_backend(
                backend_path,
                {
                    "initial_candidate": {
                        "candidate_id": "initial-bad",
                        "summary": "keep the incorrect implementation",
                        "edits": [{"path": "demo.py", "new_content": "def add(a, b):\n    return a - b\n"}],
                    },
                    "repair_candidates": [
                        {
                            "candidate_id": "repair-good",
                            "summary": "repair add() after failing assertion",
                            "supported_failure_classes": ["assertion_failure", "behavior_mismatch"],
                            "edits": [{"path": "demo.py", "new_content": "def add(a, b):\n    return a + b\n"}],
                        }
                    ],
                },
            )

            payload = solve_task(
                TaskRequest(
                    task_text="Fix add()",
                    file_hints=("demo.py",),
                    checks=("unittest:discover -s . -p test_demo.py -v", "compileall:demo.py"),
                    workspace_root=td,
                ),
                backend_script_path=backend_path,
            )

            self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
            self.assertEqual(payload["final_origin"], "repaired")
            self.assertEqual(payload["optimization_status"], "not_attempted")
            self.assertEqual(len(payload["attempts"]), 2)
            self.assertEqual(payload["attempts"][0]["critique"]["failure_class"], "assertion_failure")
            self.assertEqual(payload["attempts"][1]["origin"], "repaired")
            with open(module_path, "r", encoding="utf-8") as handle:
                self.assertIn("return a + b", handle.read())

    def test_retry_budget_exhaustion_is_reported_as_verification_failed(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
            backend_path = os.path.join(td, "repair_fail.json")
            with open(module_path, "w", encoding="utf-8") as handle:
                handle.write("def add(a, b):\n    return a - b\n")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import unittest\n"
                    "from demo import add\n\n"
                    "class DemoTests(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(2, 3), 5)\n"
                )
            bad_edit = {"path": "demo.py", "new_content": "def add(a, b):\n    return a - b\n"}
            write_backend(
                backend_path,
                {
                    "initial_candidate": {
                        "candidate_id": "initial-bad",
                        "summary": "leave bug in place",
                        "edits": [bad_edit],
                    },
                    "repair_candidates": [
                        {"candidate_id": "repair-1", "summary": "still wrong", "edits": [bad_edit]},
                        {"candidate_id": "repair-2", "summary": "still wrong", "edits": [bad_edit]},
                        {"candidate_id": "repair-3", "summary": "still wrong", "edits": [bad_edit]},
                    ],
                },
            )

            payload = solve_task(
                TaskRequest(
                    task_text="Fix add()",
                    file_hints=("demo.py",),
                    checks=("unittest:discover -s . -p test_demo.py -v",),
                    workspace_root=td,
                ),
                backend_script_path=backend_path,
            )

            self.assertEqual(payload["status"], SOLVE_STATUS_FAILED)
            self.assertEqual(payload["final_origin"], "none")
            self.assertEqual(payload["retry_budget"], 3)
            self.assertEqual(len(payload["attempts"]), 4)
            with open(module_path, "r", encoding="utf-8") as handle:
                self.assertIn("return a - b", handle.read())

    def test_optimizer_can_keep_a_verified_green_candidate(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
            backend_path = os.path.join(td, "optimizer_keep.json")
            with open(module_path, "w", encoding="utf-8") as handle:
                handle.write("def add(a, b):\n    return a + b\n")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import unittest\n"
                    "from demo import add\n\n"
                    "class DemoTests(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(2, 3), 5)\n"
                )
            write_backend(
                backend_path,
                {
                    "initial_candidate": {
                        "candidate_id": "initial-good",
                        "summary": "preserve correct behavior",
                        "edits": [{"path": "demo.py", "new_content": "def add(a, b):\n    return a + b\n"}],
                    },
                    "optimization_candidate": {
                        "candidate_id": "optimize-style",
                        "summary": "add a harmless docstring",
                        "edits": [{"path": "demo.py", "new_content": 'def add(a, b):\n    """Return the sum of two integers."""\n    return a + b\n'}],
                    },
                },
            )

            payload = solve_task(
                TaskRequest(
                    task_text="Polish add()",
                    file_hints=("demo.py",),
                    checks=("unittest:discover -s . -p test_demo.py -v", "compileall:demo.py"),
                    workspace_root=td,
                ),
                backend_script_path=backend_path,
                optimize=True,
            )

            self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
            self.assertEqual(payload["optimization_status"], "attempted_and_kept")
            self.assertEqual(payload["final_origin"], "optimized")
            self.assertEqual(payload["attempts"][-1]["phase"], "optimization")
            with open(module_path, "r", encoding="utf-8") as handle:
                self.assertIn('"""Return the sum of two integers."""', handle.read())

    def test_optimizer_rollback_keeps_last_verified_good_state(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
            backend_path = os.path.join(td, "optimizer_rollback.json")
            with open(module_path, "w", encoding="utf-8") as handle:
                handle.write("def add(a, b):\n    return a + b\n")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import unittest\n"
                    "from demo import add\n\n"
                    "class DemoTests(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(2, 3), 5)\n"
                )
            write_backend(
                backend_path,
                {
                    "initial_candidate": {
                        "candidate_id": "initial-good",
                        "summary": "preserve correct behavior",
                        "edits": [{"path": "demo.py", "new_content": "def add(a, b):\n    return a + b\n"}],
                    },
                    "optimization_candidate": {
                        "candidate_id": "optimize-bad",
                        "summary": "introduce a regression",
                        "edits": [{"path": "demo.py", "new_content": "def add(a, b):\n    return a - b\n"}],
                    },
                },
            )

            payload = solve_task(
                TaskRequest(
                    task_text="Polish add()",
                    file_hints=("demo.py",),
                    checks=("unittest:discover -s . -p test_demo.py -v",),
                    workspace_root=td,
                ),
                backend_script_path=backend_path,
                optimize=True,
            )

            self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
            self.assertEqual(payload["optimization_status"], "attempted_and_rolled_back")
            self.assertEqual(payload["final_origin"], "initial")
            with open(module_path, "r", encoding="utf-8") as handle:
                self.assertIn("return a + b", handle.read())

    def test_exact_symbolic_tasks_bypass_repair_loop(self):
        payload = solve_task(TaskRequest(task_text='count substring "ana" in "banana"'))
        self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
        self.assertEqual(payload["final_origin"], "deterministic_exact_tool")
        self.assertEqual(payload["attempts"], [])

    def test_no_backend_still_fails_closed_for_coding_tasks(self):
        payload = solve_task(
            TaskRequest(
                task_text="Fix demo.py",
                file_hints=("demo.py",),
                checks=("compileall:demo.py",),
            )
        )
        self.assertEqual(payload["status"], SOLVE_STATUS_BLOCKED)
        self.assertEqual(payload["attempts"], [])
