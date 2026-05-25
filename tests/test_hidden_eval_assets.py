import json
import os
import tempfile
import unittest

from eval_harness.hidden_eval_assets import (
    CATEGORY_NAMES,
    HOLDOUT_REGISTRY_PATH,
    PRIVATE_TARGETS_PATH,
    SEED_SET_PATH,
    SPEC_JSON_PATH,
    SPEC_MD_PATH,
    load_hidden_eval_holdout_registry,
    load_hidden_eval_private_targets,
    load_hidden_eval_seed_set,
    load_hidden_eval_spec,
    validate_hidden_eval_assets,
)
from eval_harness.hidden_seed_runner import run_hidden_eval_seed_execution
from eval_harness.small_model_comparison import build_small_model_smoke_comparison


class HiddenEvalAssetTests(unittest.TestCase):
    def test_hidden_eval_artifacts_exist_and_validate(self):
        report = validate_hidden_eval_assets()
        self.assertEqual(report["schema"], "hidden_eval_asset_validation_v1")
        self.assertEqual(report["item_count"], 9)
        self.assertGreater(report["split_counts"]["frozen"], 0)
        self.assertGreater(report["split_counts"]["rotating"], 0)

    def test_seed_set_covers_all_categories_with_unique_ids(self):
        items = load_hidden_eval_seed_set()
        self.assertEqual(len({item["id"] for item in items}), len(items))
        self.assertEqual({item["category"] for item in items}, set(CATEGORY_NAMES))

    def test_registry_matches_seed_item_count_and_split_summary(self):
        items = load_hidden_eval_seed_set()
        registry = load_hidden_eval_holdout_registry()
        self.assertEqual(registry["schema"], "hidden_eval_holdout_hash_registry_v1")
        self.assertEqual(len(registry["items"]), len(items))
        self.assertGreater(registry["split_summary"]["frozen"], 0)
        self.assertGreater(registry["split_summary"]["rotating"], 0)

    def test_private_targets_cover_all_seed_items(self):
        items = load_hidden_eval_seed_set()
        private_targets = load_hidden_eval_private_targets()
        for item in items:
            self.assertIn(item["private_target_ref"], private_targets)

    def test_spec_and_artifacts_do_not_claim_public_benchmark_scores(self):
        texts = [
            SPEC_MD_PATH.read_text(encoding="utf-8"),
            SPEC_JSON_PATH.read_text(encoding="utf-8"),
            SEED_SET_PATH.read_text(encoding="utf-8"),
            HOLDOUT_REGISTRY_PATH.read_text(encoding="utf-8"),
            PRIVATE_TARGETS_PATH.read_text(encoding="utf-8"),
        ]
        combined = "\n".join(texts).lower()
        self.assertNotIn("leaderboard", combined)
        self.assertNotIn("public benchmark score", combined)
        self.assertNotIn("1x/10x/100x", combined)

    def test_machine_readable_spec_has_required_categories(self):
        spec = load_hidden_eval_spec()
        self.assertEqual(spec["schema"], "hidden_eval_spec_v1")
        self.assertEqual(set(spec["categories"].keys()), set(CATEGORY_NAMES))
        self.assertIn("scoring_rules", spec)
        self.assertGreater(len(spec["scoring_rules"]), 0)

    def test_hidden_eval_seed_runner_exact_item_bypasses_backend(self):
        with tempfile.TemporaryDirectory() as td:
            report = run_hidden_eval_seed_execution(
                seed_ids=("hidden_exact_001",),
                workspace_root=".",
                backend_config_path=os.path.join(td, "missing_qwen_config.json"),
                output_root=os.path.join(td, "hidden_eval_runs"),
                use_default_agent_report_dir=False,
            )

        self.assertEqual(report["schema"], "hidden_eval_seed_run_v1")
        self.assertEqual(report["counts"]["total_attempted"], 1)
        self.assertEqual(report["counts"]["passed"], 1)
        self.assertEqual(report["counts"]["exact_tool_routed"], 1)
        self.assertEqual(report["counts"]["backend_routed"], 0)
        self.assertFalse(report["private_target_leakage"])

    def test_hidden_eval_seed_runner_routes_retrieval_through_citation_verifier(self):
        with tempfile.TemporaryDirectory() as td:
            report = run_hidden_eval_seed_execution(
                seed_ids=("hidden_retrieval_001",),
                workspace_root=".",
                backend_config_path=os.path.join(td, "missing_qwen_config.json"),
                output_root=os.path.join(td, "hidden_eval_runs"),
                use_default_agent_report_dir=False,
            )

        self.assertEqual(report["counts"]["retrieval_routed"], 1)
        self.assertEqual(report["counts"]["retrieval_supported"], 1)
        self.assertEqual(report["counts"]["citation_verifier_passed"], 1)
        self.assertEqual(report["items"][0]["final_status"], "passed")
        self.assertTrue(report["items"][0]["private_target_used"])
        self.assertFalse(report["items"][0]["private_target_exposed_in_public_summary"])

    def test_hidden_eval_public_summary_does_not_expose_private_targets(self):
        with tempfile.TemporaryDirectory() as td:
            report = run_hidden_eval_seed_execution(
                seed_ids=("hidden_exact_001", "hidden_retrieval_001"),
                workspace_root=".",
                backend_config_path=os.path.join(td, "missing_qwen_config.json"),
                output_root=os.path.join(td, "hidden_eval_runs"),
                use_default_agent_report_dir=False,
            )

        summary_text = json.dumps(report, sort_keys=True)
        private_targets = load_hidden_eval_private_targets()
        self.assertNotIn(private_targets["hidden_exact_001"]["expected_output"], summary_text)
        self.assertNotIn("expected_output", summary_text)
        self.assertNotIn("behavior_assertions", summary_text)
        self.assertFalse(report["private_target_exposed_in_public_summary"])

    def test_hidden_eval_coding_item_can_run_through_scripted_backend(self):
        fixed_math_ops = (
            "def bounded_average(values, lower=0.0, upper=100.0):\n"
            "    if not values:\n"
            "        raise ValueError(\"values must not be empty\")\n"
            "    avg = sum(values) / len(values)\n"
            "    if avg < lower:\n"
            "        return lower\n"
            "    if avg > upper:\n"
            "        return upper\n"
            "    return avg\n"
        )
        with tempfile.TemporaryDirectory() as td:
            script_path = os.path.join(td, "candidate.json")
            with open(script_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_id": "scripted-hidden-fix",
                        "summary": "repair bounded average",
                        "edits": [{"path": "math_ops.py", "new_content": fixed_math_ops}],
                    },
                    handle,
                )
            report = run_hidden_eval_seed_execution(
                seed_ids=("hidden_code_patch_001",),
                workspace_root=".",
                backend_script_path=script_path,
                output_root=os.path.join(td, "hidden_eval_runs"),
                use_default_agent_report_dir=False,
            )

        self.assertEqual(report["counts"]["backend_routed"], 1)
        self.assertEqual(report["counts"]["verifier_reached"], 1)
        self.assertEqual(report["counts"]["verifier_passed"], 1)
        self.assertEqual(report["items"][0]["final_status"], "passed")
        self.assertEqual(report["items"][0]["backend_kind"], "scripted")
        self.assertTrue(report["items"][0]["agent_run_id"])

    def test_small_model_comparison_report_has_no_winner_language(self):
        baseline = {
            "backend_kind": "local_transformers_in_process",
            "model_id_or_path": "./run_artifacts/local_models/qwen2.5-coder-0.5b-instruct",
            "counts": {"backend_routed": 2, "verifier_reached": 1, "verifier_passed": 0, "blocked": 1, "failed": 1},
        }
        candidate = {
            "backend_kind": "local_transformers_in_process",
            "model_id_or_path": "./run_artifacts/local_models/qwen2.5-coder-1.5b-instruct",
            "counts": {"backend_routed": 2, "verifier_reached": 2, "verifier_passed": 1, "blocked": 0, "failed": 1},
        }
        report = build_small_model_smoke_comparison(
            baseline_label="qwen2.5-coder-0.5b",
            baseline_hidden_eval_summary=baseline,
            candidate_label="qwen2.5-coder-1.5b",
            candidate_hidden_eval_summary=candidate,
            baseline_backend_smoke={"model_load_status": "passed", "generation_status": "passed"},
            candidate_backend_smoke={"model_load_status": "passed", "generation_status": "passed"},
        )

        self.assertEqual(report["schema"], "small_model_smoke_comparison_v1")
        self.assertIsNone(report["winner_selected"])
        self.assertFalse(report["bakeoff_executed"])
        self.assertEqual(report["quality_claim"], "none")
        self.assertFalse(report["proves_model_quality"])
        self.assertEqual(report["candidate"]["hidden_verifier_passed"], 1)


if __name__ == "__main__":
    unittest.main()
