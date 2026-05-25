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
    OPTIONAL_3B_CANDIDATE_MODEL_ID,
    SMALL_CANDIDATE_MODEL_OPTIONS,
    probe_optional_qwen_3b_readiness,
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


def fake_snapshot_download(seen=None, fail=False):
    seen = seen if seen is not None else []

    def _fake(*, model_id, destination_path, source_schema, source_payload):
        seen.append((model_id, str(destination_path), source_schema, source_payload))
        if fail:
            raise OSError("network unavailable")
        target = Path(destination_path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")
        (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"fake-model")
        (target / "backend_model_source.json").write_text(
            json.dumps(
                {
                    "schema": source_schema,
                    "model_id": model_id,
                    "quality_claim": source_payload.get("quality_claim"),
                }
            ),
            encoding="utf-8",
        )

    return _fake


class SmallCandidateProvisionTests(unittest.TestCase):
    def test_autonomous_provision_uses_first_priority_small_model(self):
        seen = []
        with tempfile.TemporaryDirectory(dir=".") as td:
            destination = os.path.join(td, "smollm")
            config_path = os.path.join(td, "backend_config.json")
            fake_module = fake_transformers_download_module(seen=seen)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with patch("agent.backend._huggingface_model_safety_metadata", return_value={
                    "metadata_lookup_status": "passed",
                    "hub_private": False,
                    "hub_gated": False,
                    "license": "apache-2.0",
                }):
                    with patch("agent.backend._snapshot_download_model_files", side_effect=fake_snapshot_download(seen=seen)):
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
        self.assertEqual(seen[0][0], SMALL_CANDIDATE_MODEL_OPTIONS[0]["model_id"])
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

    def test_qwen_candidate_can_be_selected_explicitly_and_writes_qwen_config(self):
        seen = []
        with tempfile.TemporaryDirectory(dir=".") as td:
            destination = os.path.join(td, "qwen")
            config_path = os.path.join(td, "qwen_config.json")
            fake_module = fake_transformers_download_module(seen=seen)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with patch("agent.backend._huggingface_model_safety_metadata", return_value={
                    "metadata_lookup_status": "passed",
                    "hub_private": False,
                    "hub_gated": False,
                    "license": "apache-2.0",
                }):
                    with patch("agent.backend._snapshot_download_model_files", side_effect=fake_snapshot_download(seen=seen)):
                        report = provision_small_candidate_model(
                            model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                            destination=destination,
                            config_path=config_path,
                        )

            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)

        self.assertTrue(report["ok"])
        self.assertEqual(report["selected_model_id"], "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertEqual(seen[0][0], "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertIn("qwen", config["agent"]["backend_model_id_or_path"].lower())
        self.assertTrue(config["agent"]["backend_local_files_only"])
        self.assertFalse(report["proves_model_quality"])

    def test_qwen_static_smoke_config_is_local_only_and_non_default(self):
        path = os.path.join("configs", "backend_smoke_qwen2_5_coder_0_5b.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        agent = payload["agent"]
        self.assertEqual(agent["backend_kind"], "local_transformers_in_process")
        self.assertIn("qwen2.5-coder-0.5b-instruct", agent["backend_model_id_or_path"])
        self.assertTrue(agent["backend_local_files_only"])
        self.assertFalse(agent["backend_trust_remote_code"])
        self.assertLessEqual(agent["backend_max_new_tokens"], 512)

    def test_qwen_1_5b_candidate_can_be_selected_explicitly(self):
        seen = []
        with tempfile.TemporaryDirectory(dir=".") as td:
            destination = os.path.join(td, "qwen_1_5b")
            config_path = os.path.join(td, "qwen_1_5b_config.json")
            fake_module = fake_transformers_download_module(seen=seen)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with patch("agent.backend._huggingface_model_safety_metadata", return_value={
                    "metadata_lookup_status": "passed",
                    "hub_private": False,
                    "hub_gated": False,
                    "license": "apache-2.0",
                }):
                    with patch("agent.backend._snapshot_download_model_files", side_effect=fake_snapshot_download(seen=seen)):
                        report = provision_small_candidate_model(
                            model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                            destination=destination,
                            config_path=config_path,
                        )

            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)

        self.assertTrue(report["ok"])
        self.assertEqual(report["selected_model_id"], "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self.assertEqual(seen[0][0], "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self.assertTrue(report["internet_used"])
        self.assertIn("qwen_1_5b", config["agent"]["backend_model_id_or_path"].lower())
        self.assertFalse(report["proves_model_quality"])

    def test_qwen_1_5b_static_smoke_config_is_local_only(self):
        path = os.path.join("configs", "backend_smoke_qwen2_5_coder_1_5b.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        agent = payload["agent"]
        self.assertEqual(agent["backend_kind"], "local_transformers_in_process")
        self.assertIn("qwen2.5-coder-1.5b-instruct", agent["backend_model_id_or_path"])
        self.assertTrue(agent["backend_local_files_only"])
        self.assertFalse(agent["backend_trust_remote_code"])
        self.assertLessEqual(agent["backend_max_new_tokens"], 512)

    def test_optional_3b_probe_is_guarded_and_does_not_download(self):
        with tempfile.TemporaryDirectory(dir=".") as td:
            report = probe_optional_qwen_3b_readiness(destination=os.path.join(td, "missing_3b"))

        self.assertEqual(report["model_id"], OPTIONAL_3B_CANDIDATE_MODEL_ID)
        self.assertEqual(report["status"], "not_attempted_due_to_hardware_or_policy")
        self.assertFalse(report["download_attempted"])
        self.assertFalse(report["download_allowed_by_default"])
        self.assertEqual(report["quality_claim"], "none")

    def test_download_failures_are_reported_without_fake_success(self):
        with tempfile.TemporaryDirectory(dir=".") as td:
            fake_module = fake_transformers_download_module(fail=True)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with patch("agent.backend._huggingface_model_safety_metadata", return_value={
                    "metadata_lookup_status": "passed",
                    "hub_private": False,
                    "hub_gated": False,
                    "license": "apache-2.0",
                }):
                    with patch("agent.backend._snapshot_download_model_files", side_effect=fake_snapshot_download(fail=True)):
                        report = provision_small_candidate_model(
                            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
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
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            },
        )


if __name__ == "__main__":
    unittest.main()
