import json
import os
import tempfile
import unittest

import numpy as np
import torch

from download_data import (
    build_benchmark_source_risk_report,
    build_dataset_source_manifest,
    build_exact_chunk_dedup_report,
)
from train import run_hardware_readiness_validation, run_short_real_data_validation
from train import (
    _classify_gpu_evidence,
    _classify_gpu_fit_attempt,
    _parse_nvidia_smi_csv,
    _repo_local_python_report,
    run_gpu_constrained_fit_validation,
)


class HardwareEvidenceTests(unittest.TestCase):
    def test_hardware_readiness_report_is_evidence_not_training_claim(self):
        report = run_hardware_readiness_validation()

        self.assertEqual(report["schema"], "hardware_readiness_v1")
        self.assertIn("cuda_available", report)
        self.assertIn("nvidia_smi", report)
        self.assertIn("gpu_tensor_ops", report)
        self.assertIn(
            report["gpu_result_classification"],
            {
                "gpu_unavailable",
                "gpu_detected_but_unusable",
                "gpu_tiny_step_passed",
                "gpu_tiny_step_oom",
                "gpu_partial_unverified",
            },
        )
        self.assertIn("validation", report)
        self.assertEqual(report["training_readiness_claim"], "none")
        self.assertEqual(report["quality_claim"], "none")
        self.assertFalse(report["hardware_ready_for_training"])
        self.assertFalse(report["rtx_2050_4gb_realism"]["default_training_path_ready"])

    def test_nvidia_smi_csv_parser_reports_gpu_facts_without_torch_claim(self):
        parsed = _parse_nvidia_smi_csv(
            "NVIDIA GeForce RTX 2050, 4096, 581.57, 8.6\n"
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "NVIDIA GeForce RTX 2050")
        self.assertEqual(parsed[0]["total_memory_mib"], 4096)
        self.assertEqual(parsed[0]["total_memory_gb"], 4.0)
        self.assertEqual(parsed[0]["compute_capability"], "8.6")

    def test_gpu_classification_distinguishes_system_gpu_from_torch_cuda(self):
        self.assertEqual(
            _classify_gpu_evidence(
                cuda_available=False,
                system_gpu_detected=True,
                tensor_op_success=False,
                validation={"status": "pass", "device_type": "cpu"},
            ),
            "gpu_detected_but_unusable",
        )
        self.assertEqual(
            _classify_gpu_evidence(
                cuda_available=True,
                system_gpu_detected=True,
                tensor_op_success=True,
                validation={"status": "pass", "device_type": "cuda"},
            ),
            "gpu_tiny_step_passed",
        )

    def test_gpu_fit_classification_has_no_fake_success_state(self):
        self.assertEqual(
            _classify_gpu_fit_attempt(
                {
                    "status": "pass",
                    "forward_passed": True,
                    "backward_optimizer_passed": True,
                    "checkpoint_write_passed": True,
                    "resume_reload_passed": True,
                }
            ),
            "fits_mechanics_only",
        )
        self.assertEqual(
            _classify_gpu_fit_attempt(
                {
                    "status": "fail",
                    "error_type": "cuda_oom",
                    "forward_passed": False,
                }
            ),
            "does_not_fit",
        )
        self.assertEqual(
            _classify_gpu_fit_attempt(
                {
                    "status": "pass",
                    "forward_passed": True,
                    "backward_optimizer_passed": True,
                    "checkpoint_write_passed": False,
                    "resume_reload_passed": False,
                }
            ),
            "partially_unverified",
        )

    def test_repo_local_interpreter_report_is_explicit(self):
        report = _repo_local_python_report(
            repo_root=r"E:\demo_repo",
            executable=r"E:\demo_repo\.venv\Scripts\python.exe",
        )

        self.assertTrue(report["is_repo_local_venv"])
        self.assertTrue(report["source_of_truth"].endswith(r".venv\Scripts\python.exe"))

    def test_gpu_fit_report_structure_when_cuda_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        if not _repo_local_python_report()["is_repo_local_venv"]:
            self.skipTest("repo-local venv interpreter is not active")

        with tempfile.TemporaryDirectory() as td:
            report = run_gpu_constrained_fit_validation(
                checkpoint_root=os.path.join(td, "gpu_fit"),
                matrix_candidates=[
                    {
                        "name": "test_micro_fp16_32d_1l_s8",
                        "dtype_name": "float16",
                        "dim": 32,
                        "n_layers": 1,
                        "n_heads": 4,
                        "n_kv_heads": 2,
                        "ffn_dim": 64,
                        "seq_len": 8,
                        "batch_size": 1,
                    }
                ],
                run_short_validation=False,
            )

        self.assertEqual(report["schema"], "gpu_constrained_fit_validation_v1")
        self.assertEqual(report["hardware_classification"], "validation_only_for_current_repo_target")
        self.assertEqual(report["pretraining_readiness_changed"], "no")
        self.assertEqual(report["quality_claim"], "none")
        self.assertEqual(len(report["matrix"]), 1)
        self.assertIn(
            report["matrix"][0]["fit_classification"],
            {"fits_mechanics_only", "does_not_fit", "partially_unverified"},
        )
        self.assertNotEqual(report["default_training_fit"]["fit_classification"], "fits_short_validation")


class DataGovernanceEvidenceTests(unittest.TestCase):
    def test_source_manifest_marks_license_unknown_without_clearance(self):
        report = build_dataset_source_manifest()

        self.assertEqual(report["schema"], "dataset_source_governance_v1")
        self.assertGreater(report["source_count"], 0)
        self.assertEqual(report["known_license_count"], 0)
        self.assertEqual(report["legal_clearance_claim"], "none")
        self.assertIn("needs_manual_review", report["governance_classification_counts"])
        self.assertIn("excluded_from_training", report["governance_classification_counts"])
        self.assertTrue(
            all(item["license_status"] == "unknown" for item in report["sources"])
        )
        self.assertTrue(
            all(item["legal_clearance_claim"] == "none" for item in report["sources"])
        )

    def test_exact_chunk_dedup_reports_bounded_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            token_path = os.path.join(td, "tokens.bin")
            np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.uint16).tofile(token_path)
            manifest_path = os.path.join(td, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "token_artifacts_manifest_v1",
                        "artifacts": [
                            {
                                "path": token_path,
                                "source": "demo/source",
                                "subset": None,
                                "artifact_split": "train",
                                "dtype": "uint16",
                                "token_count": 8,
                            }
                        ],
                    },
                    handle,
                )

            report = build_exact_chunk_dedup_report(
                manifest_path=manifest_path,
                chunk_tokens=4,
                max_artifacts=1,
                max_chunks_per_artifact=2,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["scope"], "full_exact_chunk_scan")
        self.assertTrue(report["full_exact_chunk_scan"])
        self.assertEqual(report["chunks_inspected"], 2)
        self.assertEqual(report["chunks_available_estimate"], 2)
        self.assertEqual(report["duplicate_chunks"], 1)
        self.assertEqual(report["duplicates_by_source"]["demo/source"], 1)
        self.assertEqual(report["near_dedup_claim"], "none")

    def test_benchmark_source_risk_does_not_claim_contamination_scan(self):
        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "token_artifacts_manifest_v1",
                        "artifacts": [
                            {
                                "path": os.path.join(td, "missing.bin"),
                                "source": "openai/gsm8k",
                                "subset": "main",
                                "artifact_split": "train",
                                "token_count": 10,
                            }
                        ],
                    },
                    handle,
                )

            report = build_benchmark_source_risk_report(
                manifest_path=manifest_path,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["source_artifact_risk_count"], 1)
        self.assertEqual(report["contamination_claim"], "none")
        self.assertEqual(report["source_level_risk"]["claim"], "source_level_risk_only")
        self.assertEqual(report["content_level_overlap"]["claim"], "unverified")
        self.assertEqual(
            report["content_overlap_status"],
            "unverified_no_local_benchmark_text_or_hashes",
        )


class ShortValidationEvidenceTests(unittest.TestCase):
    def test_short_real_token_validation_runs_multiple_steps(self):
        token_dir = "./data_cache/tokens"
        if not os.path.isdir(token_dir) or not any(
            name.endswith(".bin") for name in os.listdir(token_dir)
        ):
            self.skipTest("local token artifacts are unavailable")

        old_manifest_policy = os.environ.get("MYLLM_WRITE_CHECKPOINT_MANIFEST")
        os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = "0"
        try:
            with tempfile.TemporaryDirectory() as td:
                report = run_short_real_data_validation(
                    token_cache_dir=token_dir,
                    checkpoint_dir=os.path.join(td, "checkpoints"),
                    seq_len=8,
                    pre_resume_steps=2,
                    post_resume_steps=1,
                    deterministic=True,
                )
        finally:
            if old_manifest_policy is None:
                os.environ.pop("MYLLM_WRITE_CHECKPOINT_MANIFEST", None)
            else:
                os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = old_manifest_policy

        self.assertTrue(report["ok"])
        self.assertEqual(report["run_classification"], "bounded_validation_not_pretraining")
        self.assertEqual(report["total_optimizer_steps"], 3)
        self.assertEqual(report["resumed_step"], 2)
        self.assertTrue(report["resume_success"])
        self.assertEqual(report["quality_claim"], "none")
        self.assertEqual(report["eval_gate_on_produced_checkpoint"], "checkpoint_found")


if __name__ == "__main__":
    unittest.main()
