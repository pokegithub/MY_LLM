import os
import tempfile
import unittest

from config import EvalConfig, _restore_runtime_state, runtime_config_dict
from eval.benchmark_harness import build_scale_comparison
from eval_suite import (
    EvalIntegrityError,
    _choose_eval_set,
    inspect_checkpoint_path,
    run_evaluation,
    summarize_eval_run_mode,
)


class EvalIntegrityTests(unittest.TestCase):
    def test_missing_checkpoint_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "no_checkpoint_here")
            meta = inspect_checkpoint_path(missing)

        self.assertFalse(meta["exists"])
        self.assertIsNone(meta["selected_checkpoint"])
        self.assertFalse(meta["selected_checkpoint_exists"])
        self.assertEqual(meta["load_status"], "missing")
        self.assertIn("not found", meta["error"])

    def test_run_evaluation_fails_without_checkpoint_report(self):
        snapshot = runtime_config_dict()
        try:
            with tempfile.TemporaryDirectory() as td:
                cfg = EvalConfig(
                    checkpoint_path=os.path.join(td, "missing_checkpoints"),
                    tokenizer_path="./tokenizer_data",
                    output_dir=td,
                )
                with self.assertRaises(EvalIntegrityError):
                    run_evaluation(cfg)
                self.assertFalse(os.path.exists(os.path.join(td, "eval_report.json")))
        finally:
            _restore_runtime_state(snapshot)

    def test_toy_fallback_metadata_cannot_pose_as_benchmark(self):
        cfg = EvalConfig(
            strict_real_benchmarks=False,
            allow_toy_fallback=True,
        )
        items, meta = _choose_eval_set(
            "math",
            real_items=[],
            fallback_items=[{"question": "1+1", "answer": "2"}],
            min_required=10,
            cfg=cfg,
        )

        self.assertEqual(len(items), 1)
        self.assertFalse(meta["used_real"])
        self.assertEqual(meta["run_mode"], "toy")
        self.assertEqual(meta["capability_evidence"], "toy_smoke_only")
        self.assertIn("not benchmark evidence", meta["warning"])

    def test_eval_run_mode_summary_marks_mixed_or_toy_as_not_real_evidence(self):
        summary = summarize_eval_run_mode(
            {
                "multiple_choice": {"run_mode": "real"},
                "math": {
                    "run_mode": "toy",
                    "warning": "Fallback/toy samples are smoke tests.",
                },
            }
        )

        self.assertEqual(summary["run_mode"], "mixed_or_toy")
        self.assertFalse(summary["real_benchmark_evidence"])
        self.assertEqual(summary["warnings"], ["Fallback/toy samples are smoke tests."])

    def test_scale_comparison_has_no_unverified_target_bands(self):
        comparison = build_scale_comparison(7.25)

        self.assertEqual(comparison["overall"], 7.25)
        self.assertEqual(comparison["comparison_mode"], "disabled_unverified")
        self.assertEqual(comparison["bands"], [])
        self.assertIn("no measured baseline evidence", comparison["warning"])
        self.assertNotIn("target_min", comparison)
        self.assertNotIn("target_max", comparison)


if __name__ == "__main__":
    unittest.main()
