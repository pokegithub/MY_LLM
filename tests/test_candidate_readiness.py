import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import agent_cfg
from model_selection.readiness import candidate_readiness_smoke, load_candidate_shortlist
from tests.helpers.fake_transformers import fake_transformers_sequence


def valid_candidate_json():
    return json.dumps(
        {
            "candidate_id": "readiness-fix",
            "summary": "make the smoke target return expected value",
            "edits": [
                {
                    "path": "backend_smoke_target.py",
                    "new_content": "def value():\n    return 2\n",
                }
            ],
        }
    )


class CandidateReadinessTests(unittest.TestCase):
    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_shortlist(self, directory: str, model_path: str, candidate_id: str = "local_candidate") -> str:
        payload = {
            "schema": "candidate_shortlist_v1",
            "version": "test",
            "winner_selected": False,
            "dense_only_required": True,
            "moe_allowed": False,
            "candidate_groups": [
                {
                    "candidate_id": "custom",
                    "candidate_role": "current_custom_base_candidate",
                    "dense_or_moe": "dense",
                    "custom_base_qualification_status": "requires_real_checkpoint",
                },
                {
                    "candidate_id": candidate_id,
                    "candidate_role": "open_dense_7b_class_candidate",
                    "size_class": "7b_class",
                    "dense_or_moe": "dense",
                    "local_model_path": model_path,
                    "local_availability_status": "test_local_available",
                    "license_deployment_review_status": "test_only",
                    "custom_base_qualification_status": "not_applicable",
                },
                {
                    "candidate_id": "other",
                    "candidate_role": "open_dense_14b_class_candidate",
                    "dense_or_moe": "dense",
                    "custom_base_qualification_status": "not_applicable",
                },
            ],
        }
        path = os.path.join(directory, "shortlist.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_shortlist_artifacts_define_three_dense_slots_without_winner(self):
        payload = load_candidate_shortlist("./model_selection/candidate_shortlist_v1.json")

        self.assertEqual(payload["candidate_group_count"], 3)
        self.assertFalse(payload["winner_selected"])
        self.assertFalse(payload["moe_allowed"])
        self.assertTrue(payload["dense_only_required"])
        self.assertEqual(len(payload["candidate_groups"]), 3)
        self.assertTrue(all(item["dense_or_moe"] == "dense" for item in payload["candidate_groups"]))
        custom = payload["candidate_groups"][0]
        self.assertEqual(custom["custom_base_qualification_status"], "requires_real_checkpoint")
        self.assertIn("no_usable_checkpoint_evidenced_in_standard_checkpoint_surface", custom["disqualifier_flags"])

    def test_readiness_protocol_contains_required_gates_and_no_claims(self):
        payload = self._load_json("./model_selection/candidate_readiness_protocol_v1.json")
        gate_ids = {item["gate_id"] for item in payload["required_gates"]}

        self.assertEqual(payload["schema"], "candidate_readiness_protocol_v1")
        self.assertFalse(payload["bakeoff_executed"])
        self.assertEqual(payload["quality_claim"], "none")
        self.assertEqual(payload["bakeoff_winner_claim"], "none")
        self.assertEqual(
            gate_ids,
            {
                "model_availability_gate",
                "license_deployment_review_gate",
                "runtime_load_gate",
                "tokenizer_model_generation_gate",
                "structured_candidate_json_gate",
                "safe_path_schema_validation_gate",
                "tiny_verifier_gate",
                "trajectory_quality_non_inflation_gate",
                "hidden_eval_readiness_gate",
            },
        )

    def test_candidate_smoke_configs_are_local_only_examples(self):
        for path in (
            "./configs/backend_candidate_smoke_7b_example.json",
            "./configs/backend_candidate_smoke_14b_example.json",
        ):
            payload = self._load_json(path)
            agent = payload["agent"]
            self.assertEqual(agent["backend_kind"], "local_transformers_in_process")
            self.assertTrue(agent["backend_local_files_only"])
            self.assertFalse(agent["backend_trust_remote_code"])
            self.assertIn("run_artifacts/local_models/candidates", agent["backend_model_id_or_path"])

    def test_unavailable_candidate_is_blocked_not_quality_failed(self):
        report = candidate_readiness_smoke(
            candidate_id="open_dense_7b_candidate_slot",
            shortlist_path="./model_selection/candidate_shortlist_v1.json",
            workspace_root=".",
        )

        self.assertEqual(report["schema"], "candidate_readiness_smoke_v1")
        self.assertEqual(report["smoke_level"], "candidate_not_available")
        self.assertEqual(report["failure_class"], "candidate_not_available")
        self.assertFalse(report["candidate_valid"])
        self.assertFalse(report["proves_model_quality"])
        self.assertEqual(report["bakeoff_winner_claim"], "none")
        self.assertEqual(report["quality_claim"], "none")
        self.assertIn("not a quality failure", report["failure_reason"])

    def test_mock_candidate_can_pass_structured_readiness_and_tiny_verifier(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model"
            model_dir.mkdir()
            shortlist_path = self._write_shortlist(td, str(model_dir))
            fake_module = fake_transformers_sequence([valid_candidate_json(), valid_candidate_json()])
            old_report_dir = agent_cfg.report_dir
            try:
                agent_cfg.report_dir = os.path.join(td, "agent_reports")
                with patch.dict(sys.modules, {"transformers": fake_module}):
                    report = candidate_readiness_smoke(
                        candidate_id="local_candidate",
                        shortlist_path=shortlist_path,
                        workspace_root=td,
                    )
            finally:
                agent_cfg.report_dir = old_report_dir

        self.assertEqual(report["structured_output_parse_status"], "passed")
        self.assertTrue(report["candidate_valid"])
        self.assertEqual(report["verifier_status"], "passed")
        self.assertEqual(report["smoke_level"], "candidate_valid_verifier_passed")
        self.assertFalse(report["proves_model_quality"])
        self.assertEqual(report["quality_claim"], "none")

    def test_malformed_mock_candidate_fails_readiness_without_quality_claim(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model"
            model_dir.mkdir()
            shortlist_path = self._write_shortlist(td, str(model_dir))
            fake_module = fake_transformers_sequence(["not json", "still not json", "nope"])
            with patch.dict(sys.modules, {"transformers": fake_module}):
                report = candidate_readiness_smoke(
                    candidate_id="local_candidate",
                    shortlist_path=shortlist_path,
                    workspace_root=td,
                )

        self.assertEqual(report["smoke_level"], "parse_failed")
        self.assertEqual(report["structured_output_readiness"], "failed")
        self.assertFalse(report["candidate_valid"])
        self.assertEqual(report["failure_class"], "malformed_candidate_output")
        self.assertEqual(report["quality_claim"], "none")
        self.assertEqual(report["bakeoff_winner_claim"], "none")


if __name__ == "__main__":
    unittest.main()
