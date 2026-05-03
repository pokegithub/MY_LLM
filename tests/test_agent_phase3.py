import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

from agent.orchestrator import solve_task
from agent.retrieval import list_trajectories, search_trajectories, show_trajectory, summarize_for_context
from agent.trajectory import build_trajectory_record
from agent.types import SOLVE_STATUS_BLOCKED, SOLVE_STATUS_FAILED, SOLVE_STATUS_VERIFIED, TaskRequest


@contextmanager
def writable_tempdir():
    base_dir = os.path.join(os.getcwd(), "run_artifacts", "test_agent_phase3_workspaces")
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


class AgentPhase3TrajectoryTests(unittest.TestCase):
    def test_verified_success_writes_replayable_trajectory(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
            backend_path = os.path.join(td, "candidate.json")
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
                    "candidate_id": "fix-add",
                    "summary": "repair add()",
                    "edits": [{"path": "demo.py", "new_content": "def add(a, b):\n    return a + b\n"}],
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

            self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
            self.assertTrue(payload["trajectory_store"]["ok"])
            trajectory = show_trajectory(payload["run_id"])
            self.assertIsNotNone(trajectory)
            self.assertEqual(trajectory["status"], SOLVE_STATUS_VERIFIED)
            self.assertEqual(trajectory["quality_claim"], "verification_passed")
            self.assertEqual(trajectory["learning_claim"], "none")

    def test_blocked_no_backend_run_is_stored_as_blocked(self):
        payload = solve_task(
            TaskRequest(
                task_text="Fix demo.py",
                file_hints=("demo.py",),
                checks=("compileall:demo.py",),
            )
        )
        self.assertEqual(payload["status"], SOLVE_STATUS_BLOCKED)
        trajectory = show_trajectory(payload["run_id"])
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory["status"], SOLVE_STATUS_BLOCKED)
        self.assertIn("no coding backend configured", trajectory["blocked_reason"])

    def test_retry_exhaustion_run_is_stored_as_failed(self):
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
            trajectory = show_trajectory(payload["run_id"])
            self.assertIsNotNone(trajectory)
            self.assertEqual(trajectory["status"], SOLVE_STATUS_FAILED)
            self.assertGreaterEqual(trajectory["attempt_count"], 1)

    def test_exact_symbolic_task_trajectory_is_labeled_deterministically(self):
        payload = solve_task(TaskRequest(task_text='count substring "ana" in "banana"'))
        trajectory = show_trajectory(payload["run_id"])
        self.assertEqual(trajectory["route"], "exact_symbolic")
        self.assertEqual(trajectory["status"], SOLVE_STATUS_VERIFIED)
        self.assertEqual(trajectory["final_origin"], "deterministic_exact_tool")

    def test_retrieval_list_show_and_search_are_truthful(self):
        with writable_tempdir() as td:
            with patch("agent.trajectory.trajectory_root", return_value=td), patch("agent.retrieval.trajectory_root", return_value=td):
                success = build_trajectory_record(
                    {
                        "run_id": "solve_success",
                        "timestamp": 3,
                        "operation": "solve",
                        "request": {"task_text": "Fix add()", "file_hints": ["demo.py"], "checks": [], "workspace_root": "."},
                        "route": {"route": "coding_edit", "reason": "hinted"},
                        "backend_status": {"configured": True, "available": True},
                        "context": {"workspace_root": ".", "files_considered": ["demo.py"], "files": [], "notes": []},
                        "plan": {"route": "coding_edit", "subgoals": [], "target_files": ["demo.py"], "checks": [], "risk_level": "low", "success_criteria": [], "retry_budget": 3, "stop_conditions": []},
                        "verification": {"summary": "green"},
                        "status": "verified_success",
                        "files_touched": ["demo.py"],
                        "blocked_reason": None,
                        "quality_claim": "verification_passed",
                        "report_path": os.path.join(td, "solve_success", "report.json"),
                        "attempts": [],
                        "final_origin": "initial",
                        "optimization_status": "not_attempted",
                    }
                )
                failed = build_trajectory_record(
                    {
                        "run_id": "solve_failed",
                        "timestamp": 2,
                        "operation": "solve",
                        "request": {"task_text": "Fix add()", "file_hints": ["demo.py"], "checks": [], "workspace_root": "."},
                        "route": {"route": "coding_edit", "reason": "hinted"},
                        "backend_status": {"configured": True, "available": True},
                        "context": {"workspace_root": ".", "files_considered": ["demo.py"], "files": [], "notes": []},
                        "plan": {"route": "coding_edit", "subgoals": [], "target_files": ["demo.py"], "checks": [], "risk_level": "low", "success_criteria": [], "retry_budget": 3, "stop_conditions": []},
                        "verification": {"summary": "red"},
                        "status": "verification_failed",
                        "files_touched": ["demo.py"],
                        "blocked_reason": "verification failed",
                        "quality_claim": "none",
                        "report_path": os.path.join(td, "solve_failed", "report.json"),
                        "attempts": [{"attempt_index": 1, "phase": "initial", "origin": "initial", "candidate_id": "x", "candidate_summary": "bad", "files_touched": ["demo.py"], "verification": {"summary": "red"}, "critique": {"failure_class": "assertion_failure"}, "kept": False}],
                        "final_origin": "none",
                        "optimization_status": "not_attempted",
                    }
                )
                blocked = build_trajectory_record(
                    {
                        "run_id": "solve_blocked",
                        "timestamp": 1,
                        "operation": "solve",
                        "request": {"task_text": "Fix add()", "file_hints": ["demo.py"], "checks": [], "workspace_root": "."},
                        "route": {"route": "coding_edit", "reason": "hinted"},
                        "backend_status": {"configured": False, "available": False},
                        "context": {"workspace_root": ".", "files_considered": ["demo.py"], "files": [], "notes": []},
                        "plan": {"route": "coding_edit", "subgoals": [], "target_files": ["demo.py"], "checks": [], "risk_level": "low", "success_criteria": [], "retry_budget": 3, "stop_conditions": []},
                        "verification": {"summary": "blocked"},
                        "status": "blocked_unverified",
                        "files_touched": [],
                        "blocked_reason": "no coding backend configured",
                        "quality_claim": "none",
                        "report_path": os.path.join(td, "solve_blocked", "report.json"),
                        "attempts": [],
                        "final_origin": "none",
                        "optimization_status": "not_attempted",
                    }
                )
                for record in (success, failed, blocked):
                    run_dir = os.path.join(td, record["run_id"])
                    os.makedirs(run_dir, exist_ok=True)
                    with open(os.path.join(run_dir, "trajectory.json"), "w", encoding="utf-8") as handle:
                        json.dump(record, handle)

                listed = list_trajectories(root=td, route="coding_edit", limit=10)
                self.assertEqual(len(listed), 3)
                shown = show_trajectory("solve_success", root=td)
                self.assertEqual(shown["run_id"], "solve_success")
                ranked = search_trajectories(query="Fix add", root=td, touched_file="demo.py", limit=3)
                self.assertEqual(ranked[0]["run_id"], "solve_success")
                summaries = summarize_for_context(query="Fix add", file_hints=("demo.py",), root=td, limit=2)
                self.assertLessEqual(len(summaries), 2)
                self.assertTrue(all(len(item) <= 220 for item in summaries))
                self.assertIn("status=verified_success", summaries[0])

    def test_trajectory_write_degradation_is_explicit(self):
        with patch("agent.orchestrator.write_trajectory_record", side_effect=PermissionError("trajectory locked")):
            payload = solve_task(TaskRequest(task_text='count substring "ana" in "banana"'))
        self.assertFalse(payload["trajectory_store"]["ok"])
        self.assertTrue(payload["degraded_mode"])
        self.assertTrue(
            any(item["target"] == "trajectory_store" for item in payload["artifact_warnings"])
        )


if __name__ == "__main__":
    unittest.main()
