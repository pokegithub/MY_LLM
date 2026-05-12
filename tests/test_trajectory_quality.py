import json
import os
import tempfile
import unittest

from agent.trajectory_quality import (
    BACKENDLESS_BLOCKED,
    EXACT_TOOL_ONLY,
    FAILURE_ANALYSIS,
    FIXTURE_ONLY,
    PREFERENCE_LOSER,
    PREFERENCE_WINNER,
    RETRIEVAL_MEMORY,
    SFT_POSITIVE,
    build_quality_audit,
    classify_trajectory,
    load_quality_schema,
)


def write_trajectory(root: str, run_id: str, payload: dict) -> None:
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "trajectory.json")
    payload = dict(payload)
    payload.setdefault("schema", "agent_trajectory_v1")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def base_record(**overrides) -> dict:
    payload = {
        "run_id": "solve_real_backend",
        "timestamp": 10,
        "operation": "solve",
        "task_text": "Repair calculator add behavior in calculator.py",
        "route": "coding_edit",
        "status": "verified_success",
        "quality_claim": "verification_passed",
        "final_origin": "repaired",
        "backend_status": {
            "kind": "local_transformers_in_process",
            "configured": True,
            "available": True,
        },
        "files_touched": ["calculator.py"],
        "failure_classes": ["assertion_failure"],
        "blocked_reason": None,
        "verification_summary": "all verification checks passed",
        "attempts": [
            {
                "attempt_index": 1,
                "phase": "initial",
                "origin": "initial",
                "candidate_id": "bad",
                "candidate_summary": "kept incorrect subtraction behavior",
                "files_touched": ["calculator.py"],
                "kept": False,
                "verification": {
                    "overall_passed": False,
                    "meaningful": True,
                    "summary": "assertion failed",
                    "quality_claim": "none",
                    "checks": [],
                },
                "critique": {"failure_class": "assertion_failure"},
            },
            {
                "attempt_index": 2,
                "phase": "repair",
                "origin": "repaired",
                "candidate_id": "good",
                "candidate_summary": "restore addition behavior",
                "files_touched": ["calculator.py"],
                "kept": True,
                "verification": {
                    "overall_passed": True,
                    "meaningful": True,
                    "summary": "all verification checks passed",
                    "quality_claim": "verification_passed",
                    "checks": [],
                },
            },
        ],
        "winning_attempt": 2,
    }
    payload.update(overrides)
    return payload


class TrajectoryQualityTests(unittest.TestCase):
    def test_quality_schema_contains_required_classes(self):
        schema = load_quality_schema()
        class_ids = {item["class_id"] for item in schema["quality_classes"]}
        self.assertEqual(schema["schema"], "trajectory_quality_schema_v1")
        self.assertEqual(schema["learning_claim"], "none")
        self.assertIn(SFT_POSITIVE, class_ids)
        self.assertIn(PREFERENCE_WINNER, class_ids)
        self.assertIn(PREFERENCE_LOSER, class_ids)
        self.assertIn(RETRIEVAL_MEMORY, class_ids)
        self.assertIn(EXACT_TOOL_ONLY, class_ids)
        self.assertIn(BACKENDLESS_BLOCKED, class_ids)

    def test_verified_real_backend_coding_trace_can_be_sft_positive(self):
        item = classify_trajectory(base_record())

        self.assertTrue(item["eligible"]["sft_positive"])
        self.assertTrue(item["eligible"]["preference_winner"])
        self.assertTrue(item["eligible"]["retrieval_memory"])
        self.assertIn(SFT_POSITIVE, item["quality_classes"])
        self.assertIn("real_backend_verified_coding_trace", item["reasons"])

    def test_exact_trace_is_exact_curriculum_only_not_sft(self):
        item = classify_trajectory(
            base_record(
                run_id="solve_exact",
                task_text='count substring "ana" in "banana"',
                route="exact_symbolic",
                status="verified_success",
                final_origin="deterministic_exact_tool",
                backend_status={"kind": "deterministic_exact_tool", "configured": False, "available": False},
                attempts=[],
                files_touched=[],
                winning_attempt=None,
            )
        )

        self.assertFalse(item["eligible"]["sft_positive"])
        self.assertTrue(item["eligible"]["exact_tool_curriculum"])
        self.assertIn(EXACT_TOOL_ONLY, item["quality_classes"])

    def test_backendless_blocked_trace_is_not_positive_training_data(self):
        item = classify_trajectory(
            base_record(
                run_id="solve_blocked",
                status="blocked_unverified",
                quality_claim="none",
                final_origin="none",
                backend_status={"kind": "none", "configured": False, "available": False},
                attempts=[],
                files_touched=[],
                blocked_reason="no coding backend configured",
                winning_attempt=None,
            )
        )

        self.assertFalse(item["eligible"]["sft_positive"])
        self.assertTrue(item["eligible"]["retrieval_memory"])
        self.assertTrue(item["eligible"]["failure_analysis"])
        self.assertIn(BACKENDLESS_BLOCKED, item["quality_classes"])

    def test_failed_trace_can_be_loser_and_failure_analysis_only_with_evidence(self):
        item = classify_trajectory(
            base_record(
                run_id="solve_failed",
                status="verification_failed",
                quality_claim="none",
                final_origin="none",
                backend_status={"kind": "local_transformers_in_process", "configured": True, "available": True},
                files_touched=[],
                blocked_reason="verification failed; applied edits were rolled back",
                attempts=[
                    {
                        "attempt_index": 1,
                        "phase": "initial",
                        "kept": False,
                        "verification": {
                            "overall_passed": False,
                            "meaningful": True,
                            "summary": "assertion failed",
                        },
                        "critique": {"failure_class": "assertion_failure"},
                    }
                ],
                winning_attempt=None,
            )
        )

        self.assertFalse(item["eligible"]["sft_positive"])
        self.assertFalse(item["eligible"]["preference_winner"])
        self.assertTrue(item["eligible"]["preference_loser"])
        self.assertTrue(item["eligible"]["failure_analysis"])
        self.assertIn(PREFERENCE_LOSER, item["quality_classes"])
        self.assertIn(FAILURE_ANALYSIS, item["quality_classes"])

    def test_scripted_fixture_trace_is_rejected_for_positive_training(self):
        item = classify_trajectory(
            base_record(
                run_id="solve_fixture",
                backend_status={"kind": "scripted", "configured": True, "available": True},
                files_touched=["run_artifacts/test_agent_phase3_workspaces/case_x/demo.py"],
            )
        )

        self.assertFalse(item["eligible"]["sft_positive"])
        self.assertTrue(item["rejected_for_positive_training"])
        self.assertIn(FIXTURE_ONLY, item["quality_classes"])

    def test_audit_summary_counts_are_explicit_and_non_overclaiming(self):
        with tempfile.TemporaryDirectory() as td:
            write_trajectory(td, "solve_real_backend", base_record())
            write_trajectory(
                td,
                "solve_exact",
                base_record(
                    run_id="solve_exact",
                    task_text="arithmetic: 2 + 2",
                    route="exact_symbolic",
                    final_origin="deterministic_exact_tool",
                    backend_status={"kind": "deterministic_exact_tool", "configured": False, "available": False},
                    attempts=[],
                    files_touched=[],
                    winning_attempt=None,
                ),
            )
            write_trajectory(
                td,
                "solve_blocked",
                base_record(
                    run_id="solve_blocked",
                    status="blocked_unverified",
                    quality_claim="none",
                    final_origin="none",
                    backend_status={"kind": "none", "configured": False, "available": False},
                    attempts=[],
                    files_touched=[],
                    blocked_reason="no coding backend configured",
                    winning_attempt=None,
                ),
            )

            report = build_quality_audit(root=td, write_report=False)

        self.assertEqual(report["schema"], "trajectory_quality_audit_v1")
        self.assertEqual(report["scanned_trajectories"], 3)
        self.assertEqual(report["eligibility_counts"]["sft_positive_eligible"], 1)
        self.assertEqual(report["eligibility_counts"]["exact_tool_curriculum_only"], 1)
        self.assertEqual(report["eligibility_counts"]["retrieval_memory_eligible"], 3)
        self.assertEqual(report["learning_claim"], "none")
        self.assertEqual(report["model_improvement_claim"], "none")


if __name__ == "__main__":
    unittest.main()
