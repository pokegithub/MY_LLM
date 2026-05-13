import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.backend import (
    ALLOWED_SMALL_CANDIDATE_MODEL_IDS,
    SMALL_CANDIDATE_MODEL_OPTIONS,
    provision_small_candidate_model,
)


def fake_transformers_download_module(*, fail=False, seen=None):
    seen = seen if seen is not None else []

    class FakeTokenizer:
        def save_pretrained(self, path):
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "tokenizer.json").write_text("{}", encoding="utf-8")
            (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        def save_pretrained(self, path, safe_serialization=True):
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors").write_bytes(b"fake-model")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.append(("tokenizer", model_id, kwargs))
            if fail:
                raise OSError("network unavailable")
            return FakeTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.append(("model", model_id, kwargs))
            if fail:
                raise OSError("network unavailable")
            return FakeModel()

    return types.SimpleNamespace(
        __version__="fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )


class SmallCandidateProvisionTests(unittest.TestCase):
    def test_autonomous_provision_uses_first_priority_small_model(self):
        seen = []
        with tempfile.TemporaryDirectory(dir=".") as td:
            destination = os.path.join(td, "smollm")
            config_path = os.path.join(td, "backend_config.json")
            fake_module = fake_transformers_download_module(seen=seen)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                report = provision_small_candidate_model(
                    destination=destination,
                    config_path=config_path,
                )

            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)

        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "provisioned")
        self.assertTrue(report["downloaded_autonomously"])
        self.assertEqual(report["selected_model_id"], SMALL_CANDIDATE_MODEL_OPTIONS[0]["model_id"])
        self.assertEqual(seen[0][1], SMALL_CANDIDATE_MODEL_OPTIONS[0]["model_id"])
        self.assertTrue(config["agent"]["backend_local_files_only"])
        self.assertEqual(config["agent"]["backend_kind"], "local_transformers_in_process")
        self.assertFalse(report["proves_model_quality"])
        self.assertEqual(report["bakeoff_winner_claim"], "none")
        self.assertEqual(report["quality_claim"], "none")

    def test_unlisted_model_id_is_rejected_by_large_model_guard(self):
        report = provision_small_candidate_model(model_id="some-org/some-7b-model")

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failure_class"], "unsupported_capability")
        self.assertFalse(report["manual_action_required"])

    def test_download_failures_are_reported_without_fake_success(self):
        with tempfile.TemporaryDirectory(dir=".") as td:
            fake_module = fake_transformers_download_module(fail=True)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                report = provision_small_candidate_model(
                    destination=os.path.join(td, "candidate"),
                    config_path=os.path.join(td, "config.json"),
                )

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_class"], "all_small_candidate_options_failed")
        self.assertTrue(report["manual_action_required"])
        self.assertTrue(report["attempts"])
        self.assertTrue(all(item["status"] == "failed" for item in report["attempts"]))

    def test_allowed_small_candidate_set_remains_narrow(self):
        self.assertEqual(
            ALLOWED_SMALL_CANDIDATE_MODEL_IDS,
            {
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            },
        )


if __name__ == "__main__":
    unittest.main()
