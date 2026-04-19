import os
import tempfile
import unittest

import numpy as np

from download_data import (
    config_hash,
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
