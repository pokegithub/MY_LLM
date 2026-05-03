import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

from agent.exporters import (
    export_preference_records,
    export_retrieval_records,
    export_sft_records,
)


@contextmanager
def writable_tempdir():
    base_dir = os.path.join(os.getcwd(), "run_artifacts", "test_agent_phase4_workspaces")
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"case_{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_trajectory(root: str, run_id: str, payload: dict) -> None:
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("schema", "agent_trajectory_v1")
    payload.setdefault("_trajectory_path", os.path.join(run_dir, "trajectory.json"))
    with open(os.path.join(run_dir, "trajectory.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def verified_success_record(run_id: str = "solve_verified") -> dict:
    return {
        "run_id": run_id,
        "timestamp": 10,
        "operation": "solve",
        "task_text": "Fix add()",
        "route": "coding_edit",
        "quality_claim": "verification_passed",
        "status": "verified_success",
        "final_origin": "repaired",
        "optimization_status": "not_attempted",
        "blocked_reason": None,
        "verification_summary": "all verification checks passed",
        "files_touched": ["demo.py"],
        "failure_classes": ["assertion_failure"],
        "lessons": ["Assertion failure resolved by restoring expected return value in demo.py."],
        "context_summary": {
            "workspace_root": ".",
            "files_considered": ["demo.py", "test_demo.py"],
            "included_files": [{"path": "demo.py", "note": "explicit file hint", "size_bytes": 20}],
            "notes": [],
        },
        "plan_summary": {
            "route": "coding_edit",
            "subgoals": ["inspect targeted files", "generate one candidate patch"],
            "target_files": ["demo.py"],
            "checks": [{"check_type": "unittest", "spec": "discover -s . -p test_demo.py -v"}],
            "risk_level": "low",
            "success_criteria": ["all verification checks pass"],
            "retry_budget": 3,
            "stop_conditions": ["stop after retry budget is exhausted"],
        },
        "artifact_paths": {"agent_report": "E:\\MY_LLM\\run_artifacts\\agent\\solve_verified\\report.json"},
        "attempts": [
            {
                "attempt_index": 1,
                "phase": "initial",
                "origin": "initial",
                "candidate_id": "bad",
                "candidate_summary": "leave bug in place",
                "files_touched": ["demo.py"],
                "verification": {
                    "overall_passed": False,
                    "meaningful": True,
                    "summary": "assertion failed",
                    "quality_claim": "none",
                    "checks": [],
                },
                "critique": {
                    "failure_class": "assertion_failure",
                    "root_cause": "candidate did not satisfy asserted behavior",
                    "repair_targets": ["demo.py"],
                    "blocked_reason": None,
                    "confidence": 0.9,
                    "evidence_summary": "unit test expected add(2,3)==5",
                },
                "kept": False,
            },
            {
                "attempt_index": 2,
                "phase": "repair",
                "origin": "repaired",
                "candidate_id": "good",
                "candidate_summary": "restore expected return value",
                "files_touched": ["demo.py"],
                "verification": {
                    "overall_passed": True,
                    "meaningful": True,
                    "summary": "all verification checks passed",
                    "quality_claim": "verification_passed",
                    "checks": [],
                },
                "critique": None,
                "kept": True,
            },
        ],
        "attempt_count": 2,
        "winning_attempt": 2,
        "request": {"task_text": "Fix add()", "file_hints": ["demo.py"], "checks": [], "workspace_root": "."},
    }


def blocked_record(run_id: str = "solve_blocked") -> dict:
    return {
        "run_id": run_id,
        "timestamp": 9,
        "operation": "solve",
        "task_text": "Fix add()",
        "route": "coding_edit",
        "quality_claim": "none",
        "status": "blocked_unverified",
        "final_origin": "none",
        "optimization_status": "not_attempted",
        "blocked_reason": "no coding backend configured",
        "verification_summary": "coding backend unavailable; solve must fail closed",
        "files_touched": [],
        "failure_classes": [],
        "lessons": ["no coding backend configured"],
        "context_summary": {"workspace_root": ".", "files_considered": [], "included_files": [], "notes": []},
        "plan_summary": {"route": "coding_edit", "subgoals": [], "target_files": [], "checks": [], "risk_level": "low", "success_criteria": [], "retry_budget": 3, "stop_conditions": []},
        "artifact_paths": {"agent_report": "E:\\MY_LLM\\run_artifacts\\agent\\solve_blocked\\report.json"},
        "attempts": [],
        "attempt_count": 0,
        "winning_attempt": None,
        "request": {"task_text": "Fix add()", "file_hints": ["demo.py"], "checks": [], "workspace_root": "."},
    }


def malformed_verified_record(run_id: str = "solve_malformed") -> dict:
    record = verified_success_record(run_id=run_id)
    record["task_text"] = ""
    return record


class AgentPhase4ExportTests(unittest.TestCase):
    def test_verified_success_exports_into_sft(self):
        with writable_tempdir() as td:
            write_trajectory(td, "solve_verified", verified_success_record())
            with patch("agent.exporters.trajectory_root", return_value=td):
                payload = export_sft_records(root=td)
            self.assertEqual(payload["record_schema"], "agent_sft_export_v1")
            self.assertEqual(payload["scanned_trajectories"], 1)
            self.assertEqual(payload["exported_records"], 1)
            self.assertTrue(payload["useful_output"])
            self.assertEqual(payload["preview"][0]["learning_claim"], "none")
            self.assertEqual(payload["preview"][0]["provenance"]["source_run_id"], "solve_verified")

    def test_blocked_and_failed_do_not_become_positive_sft_examples(self):
        with writable_tempdir() as td:
            write_trajectory(td, "solve_verified", verified_success_record())
            write_trajectory(td, "solve_blocked", blocked_record())
            with patch("agent.exporters.trajectory_root", return_value=td):
                payload = export_sft_records(root=td)
            self.assertEqual(payload["exported_records"], 1)
            self.assertEqual(payload["skipped_trajectories"], 1)
            self.assertEqual(payload["skip_reason_counts"]["status_not_verified_success"], 1)

    def test_preference_pairs_are_created_only_from_verifier_justified_evidence(self):
        with writable_tempdir() as td:
            write_trajectory(td, "solve_verified", verified_success_record())
            with patch("agent.exporters.trajectory_root", return_value=td):
                payload = export_preference_records(root=td)
            self.assertEqual(payload["record_schema"], "agent_preference_export_v1")
            self.assertEqual(payload["exported_records"], 1)
            record = payload["preview"][0]
            self.assertEqual(record["winner"]["attempt_index"], 2)
            self.assertEqual(record["loser"]["attempt_index"], 1)
            self.assertEqual(record["verifier_justification"]["loser_failure_class"], "assertion_failure")

    def test_preference_export_skips_when_no_honest_pair_exists(self):
        with writable_tempdir() as td:
            exact_verified = verified_success_record(run_id="exact_verified")
            exact_verified["route"] = "exact_symbolic"
            exact_verified["final_origin"] = "deterministic_exact_tool"
            exact_verified["attempts"] = []
            exact_verified["attempt_count"] = 0
            exact_verified["winning_attempt"] = None
            exact_verified["failure_classes"] = []
            write_trajectory(td, "exact_verified", exact_verified)
            with patch("agent.exporters.trajectory_root", return_value=td):
                payload = export_preference_records(root=td)
            self.assertEqual(payload["exported_records"], 0)
            self.assertFalse(payload["useful_output"])
            self.assertEqual(payload["skip_reason_counts"]["no_verifier_justified_preference_pair"], 1)

    def test_retrieval_export_includes_compact_truthful_summaries(self):
        with writable_tempdir() as td:
            write_trajectory(td, "solve_verified", verified_success_record())
            write_trajectory(td, "solve_blocked", blocked_record())
            with patch("agent.exporters.trajectory_root", return_value=td):
                payload = export_retrieval_records(root=td)
            self.assertEqual(payload["record_schema"], "agent_retrieval_export_v1")
            self.assertEqual(payload["exported_records"], 2)
            preview = payload["preview"][0]
            self.assertIn("experience_type", preview)
            self.assertLessEqual(len(preview["retrieval_text"]), 320)
            self.assertEqual(preview["learning_claim"], "none")

    def test_export_reject_accounting_is_explicit_for_bad_provenance(self):
        with writable_tempdir() as td:
            write_trajectory(td, "solve_verified", verified_success_record())
            write_trajectory(td, "solve_malformed", malformed_verified_record())
            with patch("agent.exporters.trajectory_root", return_value=td):
                payload = export_sft_records(root=td)
            self.assertEqual(payload["exported_records"], 1)
            self.assertEqual(payload["rejected_trajectories"], 1)
            self.assertIn("missing_required_provenance:task_text", payload["reject_reason_counts"])

    def test_export_paths_are_written_even_when_zero_useful_output(self):
        with writable_tempdir() as td:
            write_trajectory(td, "solve_blocked", blocked_record())
            with patch("agent.exporters.trajectory_root", return_value=td):
                payload = export_sft_records(root=td)
            self.assertEqual(payload["exported_records"], 0)
            self.assertFalse(payload["useful_output"])
            self.assertTrue(os.path.isfile(payload["records_path"]))
            self.assertTrue(os.path.isfile(payload["report_path"]))


if __name__ == "__main__":
    unittest.main()
