import os
import tempfile
import unittest

import numpy as np

from download_data import (
    build_benchmark_source_risk_report,
    build_dataset_source_manifest,
    build_exact_chunk_dedup_report,
    config_hash,
    load_manual_source_license_metadata,
    reconstruct_token_artifact_manifest,
    token_artifact_record,
    validate_token_artifact_manifest,
    write_token_artifact_manifest,
)
from eval_harness.contamination_checks import (
    benchmark_exclusion_coverage,
    has_low_benchmark_exclusion_coverage,
    is_contaminated,
)
from tokenizer import BPETokenizer


class DataArtifactTruthfulnessTests(unittest.TestCase):
    def test_token_artifact_record_reports_existing_file_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tokens.bin")
            np.array([1, 2, 3, 4], dtype=np.uint16).tofile(path)
            digest = config_hash({"source": "demo"})

            record = token_artifact_record(
                path=path,
                source="demo/source",
                subset="subset",
                dataset_split="train",
                artifact_split="train",
                dtype=np.uint16,
                config_digest=digest,
            )

            self.assertTrue(record["exists"])
            self.assertEqual(record["dtype"], "uint16")
            self.assertEqual(record["itemsize"], 2)
            self.assertEqual(record["size_bytes"], 8)
            self.assertEqual(record["token_count"], 4)
            self.assertEqual(record["config_hash"], digest)

    def test_token_artifact_record_rejects_misaligned_binary_size(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.bin")
            with open(path, "wb") as f:
                f.write(b"abc")

            with self.assertRaisesRegex(ValueError, "not divisible"):
                token_artifact_record(
                    path=path,
                    source="demo/source",
                    subset=None,
                    dataset_split="train",
                    artifact_split="train",
                    dtype=np.uint16,
                )

    def test_write_token_artifact_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "manifest.json")
            written = write_token_artifact_manifest([], manifest_path)
            self.assertEqual(written, manifest_path)
            self.assertTrue(os.path.exists(manifest_path))

    def test_reconstruct_manifest_marks_local_cache_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            token_dir = os.path.join(td, "tokens")
            val_dir = os.path.join(td, "val_tokens")
            os.makedirs(token_dir)
            os.makedirs(val_dir)
            known = os.path.join(
                token_dir,
                "wikimedia__wikipedia__20231101.en.bin",
            )
            unknown = os.path.join(val_dir, "local__unknown.bin")
            np.array([1, 2, 3], dtype=np.uint16).tofile(known)
            np.array([4, 5], dtype=np.uint16).tofile(unknown)
            manifest_path = os.path.join(td, "token_artifacts_manifest.json")

            report = reconstruct_token_artifact_manifest(
                token_cache_dir=token_dir,
                val_cache_dir=val_dir,
                manifest_path=manifest_path,
                dtype=np.uint16,
            )

            self.assertEqual(report["artifact_count"], 2)
            self.assertEqual(report["unknown_source_count"], 1)
            self.assertTrue(os.path.exists(manifest_path))

    def test_validate_token_artifact_manifest_checks_size_and_hash_scope(self):
        with tempfile.TemporaryDirectory() as td:
            token_path = os.path.join(td, "tokens.bin")
            np.array([1, 2, 3, 4], dtype=np.uint16).tofile(token_path)
            manifest_path = os.path.join(td, "token_artifacts_manifest.json")
            write_token_artifact_manifest(
                [
                    token_artifact_record(
                        path=token_path,
                        source="demo/source",
                        subset=None,
                        dataset_split="train",
                        artifact_split="train",
                        dtype=np.uint16,
                    )
                ],
                manifest_path,
            )

            report = validate_token_artifact_manifest(
                manifest_path=manifest_path,
                hash_block_size=8,
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["artifact_count"], 1)
            self.assertEqual(report["failed_artifact_count"], 0)
            self.assertIn("sha256", report["artifacts"][0])
            self.assertEqual(report["quality_claim"], "none")


class BenchmarkExclusionCoverageTests(unittest.TestCase):
    def test_benchmark_exclusion_coverage_is_not_content_contamination(self):
        coverage = benchmark_exclusion_coverage(
            ["truthful_qa"],
            ["truthful_qa", "openai/gsm8k"],
        )
        self.assertEqual(coverage, 0.5)

        low, same_coverage = has_low_benchmark_exclusion_coverage(
            ["truthful_qa"],
            ["truthful_qa", "openai/gsm8k"],
            threshold=0.75,
        )
        self.assertTrue(low)
        self.assertEqual(same_coverage, coverage)

    def test_legacy_is_contaminated_name_remains_alias_only(self):
        self.assertEqual(
            is_contaminated([], ["truthful_qa"], threshold=0.5),
            has_low_benchmark_exclusion_coverage(
                [],
                ["truthful_qa"],
                threshold=0.5,
            ),
        )


class DataGovernanceReportTests(unittest.TestCase):
    def test_manual_source_license_metadata_covers_sources_without_clearance(self):
        metadata = load_manual_source_license_metadata()

        self.assertTrue(metadata["ok"])
        self.assertGreaterEqual(metadata["record_count"], 29)
        self.assertEqual(metadata["legal_clearance_claim"], "none")
        for item in metadata["records"]:
            self.assertEqual(item["legal_clearance_claim"], "none")
            self.assertIn(
                item["repo_policy_status"],
                {
                    "unspecified",
                    "excluded",
                    "blocked_pending_review",
                    "allowed_by_repo_policy",
                    "restricted",
                },
            )
            self.assertIn(
                item["training_blocker_level"],
                {"hard_blocker", "caution", "informational"},
            )

    def test_source_governance_keeps_license_unknown_without_repo_evidence(self):
        report = build_dataset_source_manifest()

        self.assertEqual(report["legal_clearance_claim"], "none")
        self.assertEqual(report["manual_metadata"]["matched_source_count"], report["source_count"])
        self.assertEqual(report["manual_metadata"]["coverage_ratio"], 1.0)
        self.assertEqual(report["known_license_count"], 0)
        self.assertEqual(report["documented_license_count"], 0)
        self.assertGreater(report["unknown_license_count"], 0)
        self.assertEqual(report["declared_license_counts"], {"unknown": report["source_count"]})
        self.assertIn("source_category_counts", report)
        self.assertIn("repo_policy_status_counts", report)
        self.assertEqual(report["repo_policy_status_counts"]["blocked_pending_review"], 24)
        self.assertEqual(report["repo_policy_status_counts"]["excluded"], 5)
        self.assertEqual(report["training_blocker_level_counts"], {"hard_blocker": 29})
        self.assertEqual(report["active_hard_blocker_count"], 24)
        self.assertEqual(report["allowed_by_repo_policy_sources"], [])
        self.assertTrue(report["serious_training_blocked_by_policy"])
        for item in report["sources"]:
            self.assertIn(
                item["governance_classification"],
                {
                    "known_but_unreviewed",
                    "unknown",
                    "needs_manual_review",
                    "excluded_from_training",
                    "documented_but_unreviewed",
                    "policy_allowed_but_not_legally_cleared",
                },
            )
            self.assertEqual(item["legal_clearance_claim"], "none")
            if item["declared_license"] == "unknown":
                self.assertEqual(item["license_evidence_source"], "unknown")

    def test_partial_manual_metadata_is_reported_without_fake_clearance(self):
        with tempfile.TemporaryDirectory() as td:
            metadata_path = os.path.join(td, "source_license_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as handle:
                import json

                json.dump(
                    {
                        "schema": "manual_source_license_metadata_v1",
                        "legal_clearance_claim": "none",
                        "records": [
                            {
                                "source_id": "HuggingFaceFW/fineweb-edu",
                                "subset": "sample-10BT",
                                "source_category": "web",
                                "source_domain": "general_web",
                                "declared_license": "unknown",
                                "license_evidence_source": "unknown",
                                "review_basis": "not_reviewed",
                                "governance_classification": "needs_manual_review",
                                "repo_policy_status": "blocked_pending_review",
                                "training_blocker_level": "hard_blocker",
                                "notes": "partial metadata test",
                                "legal_clearance_claim": "none",
                            }
                        ],
                    },
                    handle,
                )

            report = build_dataset_source_manifest(metadata_path=metadata_path)

        self.assertTrue(report["ok"])
        self.assertLess(report["manual_metadata"]["coverage_ratio"], 1.0)
        self.assertGreater(report["manual_metadata"]["missing_source_metadata_count"], 0)
        self.assertEqual(report["legal_clearance_claim"], "none")
        self.assertTrue(report["serious_training_blocked_by_policy"])

    def test_repo_policy_triage_is_conservative_for_unknown_licenses(self):
        report = build_dataset_source_manifest()

        active_sources = [
            item for item in report["sources"]
            if item["active_in_current_config"]
        ]
        self.assertEqual(len(active_sources), 24)
        self.assertTrue(
            all(item["repo_policy_status"] == "blocked_pending_review" for item in active_sources)
        )
        self.assertTrue(
            all(item["training_blocker_level"] == "hard_blocker" for item in active_sources)
        )
        self.assertFalse(report["allowed_by_repo_policy_sources"])
        self.assertEqual(len(report["hard_blockers"]), report["source_count"])

    def test_allowed_by_repo_policy_is_not_legal_clearance(self):
        with tempfile.TemporaryDirectory() as td:
            metadata_path = os.path.join(td, "source_license_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as handle:
                import json

                json.dump(
                    {
                        "schema": "manual_source_license_metadata_v1",
                        "legal_clearance_claim": "none",
                        "records": [
                            {
                                "source_id": "HuggingFaceFW/fineweb-edu",
                                "subset": "sample-10BT",
                                "source_category": "web",
                                "source_domain": "general_web",
                                "declared_license": "example-license-string",
                                "license_evidence_source": "manual_repo_metadata",
                                "review_basis": "manual_repo_policy_decision",
                                "governance_classification": "policy_allowed_but_not_legally_cleared",
                                "repo_policy_status": "allowed_by_repo_policy",
                                "training_blocker_level": "caution",
                                "notes": "test metadata only",
                                "legal_clearance_claim": "none",
                            }
                        ],
                    },
                    handle,
                )

            report = build_dataset_source_manifest(metadata_path=metadata_path)

        first = next(
            item for item in report["sources"]
            if item["source"] == "HuggingFaceFW/fineweb-edu"
        )
        self.assertEqual(first["repo_policy_status"], "allowed_by_repo_policy")
        self.assertEqual(
            first["governance_classification"],
            "policy_allowed_but_not_legally_cleared",
        )
        self.assertEqual(first["training_blocker_level"], "caution")
        self.assertEqual(first["legal_clearance_claim"], "none")
        self.assertEqual(report["legal_clearance_claim"], "none")

    def test_exact_dedup_scope_fields_do_not_claim_near_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            train_path = os.path.join(td, "train.bin")
            val_path = os.path.join(td, "val.bin")
            np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16).tofile(train_path)
            np.array([1, 2, 3, 4], dtype=np.uint16).tofile(val_path)
            manifest_path = os.path.join(td, "manifest.json")
            write_token_artifact_manifest(
                [
                    token_artifact_record(
                        path=train_path,
                        source="demo/source",
                        subset=None,
                        dataset_split="train",
                        artifact_split="train",
                        dtype=np.uint16,
                    ),
                    token_artifact_record(
                        path=val_path,
                        source="demo/source",
                        subset=None,
                        dataset_split="train",
                        artifact_split="validation",
                        dtype=np.uint16,
                    ),
                ],
                manifest_path,
            )

            report = build_exact_chunk_dedup_report(
                manifest_path=manifest_path,
                chunk_tokens=4,
                max_artifacts=1,
                max_chunks_per_artifact=1,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["scope"], "bounded_exact_chunk_sample")
        self.assertFalse(report["full_exact_chunk_scan"])
        self.assertEqual(report["near_dedup_claim"], "none")
        self.assertEqual(report["full_corpus_dedup_claim"], "none")
        self.assertEqual(report["train_val_exact_overlap"]["claim"], "bounded_exact_overlap_only")
        self.assertEqual(report["train_val_exact_overlap"]["overlap_chunks"], 1)

    def test_benchmark_risk_separates_source_risk_from_content_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            token_path = os.path.join(td, "tokens.bin")
            np.array([1, 2, 3, 4], dtype=np.uint16).tofile(token_path)
            manifest_path = os.path.join(td, "manifest.json")
            write_token_artifact_manifest(
                [
                    token_artifact_record(
                        path=token_path,
                        source="truthful_qa",
                        subset="generation",
                        dataset_split="validation",
                        artifact_split="train",
                        dtype=np.uint16,
                    )
                ],
                manifest_path,
            )

            report = build_benchmark_source_risk_report(manifest_path=manifest_path)

        self.assertTrue(report["ok"])
        self.assertEqual(report["source_level_risk"]["status"], "checked")
        self.assertEqual(report["source_level_risk"]["claim"], "source_level_risk_only")
        self.assertEqual(report["source_level_risk"]["risk_artifact_count"], 1)
        self.assertEqual(report["content_level_overlap"]["claim"], "unverified")
        self.assertEqual(report["contamination_claim"], "none")


class TokenizerAuditTests(unittest.TestCase):
    def test_tokenizer_audit_reports_specials_and_roundtrip_smoke(self):
        tok = BPETokenizer()
        tok.load("./tokenizer_data")

        report = tok.audit(
            {
                "prose": "Hello world.",
                "code": "def f(x):\n    return x + 1\n",
            }
        )

        self.assertEqual(report["vocab_size"], tok.vocab_size_)
        self.assertTrue(report["special_tokens_ok"])
        self.assertGreater(report["merge_count"], 0)
        self.assertEqual(len(report["samples"]), 2)
        for sample in report["samples"]:
            self.assertGreater(sample["tokens"], 0)
            self.assertGreater(sample["chars_per_token"], 0)
            self.assertTrue(sample["roundtrip_exact"])


if __name__ == "__main__":
    unittest.main()
