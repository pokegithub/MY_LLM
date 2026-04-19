import json
import math
import os
import tempfile
import unittest

import numpy as np
import torch

from config import EDGE_0_5B_GQA, TrainConfig, build_model_config
from engine.optimizer_factory import (
    AmpPolicySettings,
    OptimizerSettings,
    build_adamw,
    build_amp_policy,
)
from model import LLM, LossOutput
from train import (
    TrainingPreflightError,
    assert_training_preflight_ready,
    load_ckpt,
    run_tiny_real_data_validation,
    run_training_preflight,
    save_ckpt,
    set_training_seed,
)


def tiny_model_config(**overrides):
    values = {
        "vocab_size": 128,
        "dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 128,
        "max_seq_len": 32,
        "sliding_window": 4,
    }
    values.update(overrides)
    return build_model_config(EDGE_0_5B_GQA, **values)


def tiny_train_config(root: str, **overrides):
    values = {
        "tokenizer_path": "./tokenizer_data",
        "checkpoint_dir": os.path.join(root, "checkpoints"),
        "val_data_dir": os.path.join(root, "val_tokens"),
        "batch_size": 2,
        "seq_len": 8,
        "grad_accum": 1,
        "total_tokens": 32,
        "warmup_tokens": 4,
        "curriculum_tokens": 16,
        "compile_model": False,
        "data_num_workers": 0,
        "data_persistent_workers": False,
        "data_pin_memory": False,
        "device_non_blocking": False,
    }
    values.update(overrides)
    return TrainConfig(**values)


class TrainingPreflightTests(unittest.TestCase):
    def test_preflight_reports_missing_tokenizer_as_failure(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = tiny_train_config(
                td,
                tokenizer_path=os.path.join(td, "missing_tokenizer"),
            )
            report = run_training_preflight(
                cfg,
                tiny_model_config(),
                token_cache_dir=os.path.join(td, "tokens"),
                token_artifact_manifest_path=os.path.join(td, "manifest.json"),
                dependency_requirements_path=None,
            )

        tokenizer_check = next(
            item for item in report["checks"]
            if item["name"] == "tokenizer_loadability"
        )
        self.assertEqual(tokenizer_check["status"], "fail")
        self.assertFalse(report["ready_for_training"])
        with self.assertRaises(TrainingPreflightError):
            assert_training_preflight_ready(report)

    def test_preflight_reads_token_artifact_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            token_dir = os.path.join(td, "tokens")
            os.makedirs(token_dir)
            token_path = os.path.join(token_dir, "demo.bin")
            np.array([1, 2, 3, 4], dtype=np.uint16).tofile(token_path)
            manifest_path = os.path.join(td, "token_artifacts_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "token_artifacts_manifest_v1",
                        "artifacts": [
                            {
                                "path": token_path,
                                "exists": True,
                                "token_count": 4,
                                "dtype": "uint16",
                            }
                        ],
                    },
                    handle,
                )

            report = run_training_preflight(
                tiny_train_config(td),
                tiny_model_config(),
                token_cache_dir=token_dir,
                token_artifact_manifest_path=manifest_path,
                dependency_requirements_path=None,
            )

        cache_check = next(
            item for item in report["checks"]
            if item["name"] == "token_cache_presence"
        )
        manifest_check = next(
            item for item in report["checks"]
            if item["name"] == "token_artifact_manifest"
        )
        self.assertEqual(cache_check["status"], "pass")
        self.assertEqual(manifest_check["status"], "pass")
        self.assertEqual(manifest_check["existing_artifact_count"], 1)

    def test_preflight_reports_dependency_drift(self):
        with tempfile.TemporaryDirectory() as td:
            req_path = os.path.join(td, "requirement.txt")
            with open(req_path, "w", encoding="utf-8") as handle:
                handle.write("definitely-missing-myllm-package>=1.0.0\n")
            report = run_training_preflight(
                tiny_train_config(td),
                tiny_model_config(),
                token_cache_dir=os.path.join(td, "tokens"),
                token_artifact_manifest_path=os.path.join(td, "manifest.json"),
                dependency_requirements_path=req_path,
            )

        dep_check = next(
            item for item in report["checks"]
            if item["name"] == "dependency_environment"
        )
        self.assertEqual(dep_check["status"], "fail")
        self.assertIn("definitely-missing-myllm-package", dep_check["failed_packages"])


class TrainingRuntimeSafetyTests(unittest.TestCase):
    def test_tiny_single_batch_training_smoke(self):
        set_training_seed(123)
        cfg = tiny_model_config()
        model = LLM(cfg).train()
        opt = build_adamw(
            model,
            OptimizerSettings(
                lr=1e-3,
                weight_decay=0.0,
                use_fused=False,
            ),
        )
        amp = build_amp_policy(
            AmpPolicySettings(
                dtype_name="bfloat16",
                device_type="cpu",
                use_grad_scaler=False,
            )
        )
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))

        before = model.embed.weight.detach().clone()
        with amp.autocast():
            out = model(ids, targets=targets)
        self.assertIsInstance(out, LossOutput)
        self.assertTrue(torch.isfinite(out.total_loss.detach()))

        amp.backward(out.total_loss)
        grad_norm = amp.clip_grad_norm_(opt, model.parameters(), 1.0)
        self.assertTrue(math.isfinite(grad_norm))
        amp.step(opt)
        amp.update()
        opt.zero_grad()

        delta = (model.embed.weight.detach() - before).abs().sum().item()
        self.assertGreater(delta, 0.0)

    def test_checkpoint_save_load_round_trip_preserves_step_and_weights(self):
        old_manifest_policy = os.environ.get("MYLLM_WRITE_CHECKPOINT_MANIFEST")
        os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = "0"
        try:
            with tempfile.TemporaryDirectory() as td:
                cfg = tiny_model_config()
                train_cfg = tiny_train_config(td)
                set_training_seed(456)
                model = LLM(cfg)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

                save_ckpt(model, opt, step=7, loss=1.25, cfg=train_cfg, master=True)

                restored = LLM(cfg)
                restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
                step = load_ckpt(restored, restored_opt, train_cfg, device=0)

                self.assertEqual(step, 7)
                for key, value in model.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, restored.state_dict()[key]),
                        key,
                    )
        finally:
            if old_manifest_policy is None:
                os.environ.pop("MYLLM_WRITE_CHECKPOINT_MANIFEST", None)
            else:
                os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = old_manifest_policy

    def test_tiny_real_token_artifact_validation_path(self):
        token_dir = "./data_cache/tokens"
        if not os.path.isdir(token_dir) or not any(
            name.endswith(".bin") for name in os.listdir(token_dir)
        ):
            self.skipTest("local token artifacts are unavailable")
        old_manifest_policy = os.environ.get("MYLLM_WRITE_CHECKPOINT_MANIFEST")
        os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = "0"
        try:
            with tempfile.TemporaryDirectory() as td:
                report = run_tiny_real_data_validation(
                    token_cache_dir=token_dir,
                    checkpoint_dir=os.path.join(td, "checkpoints"),
                    seq_len=8,
                    deterministic=True,
                )

            self.assertTrue(report["ok"])
            self.assertEqual(report["resumed_step"], 1)
            self.assertEqual(report["eval_missing_checkpoint_gate"], "missing")
            self.assertEqual(report["quality_claim"], "none")
            self.assertTrue(report["checkpoint_exists"])
            self.assertIn("model_config_hash", report)
        finally:
            if old_manifest_policy is None:
                os.environ.pop("MYLLM_WRITE_CHECKPOINT_MANIFEST", None)
            else:
                os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = old_manifest_policy

    def test_tiny_real_token_validation_is_repeatable_with_seed(self):
        token_dir = "./data_cache/tokens"
        if not os.path.isdir(token_dir) or not any(
            name.endswith(".bin") for name in os.listdir(token_dir)
        ):
            self.skipTest("local token artifacts are unavailable")
        old_manifest_policy = os.environ.get("MYLLM_WRITE_CHECKPOINT_MANIFEST")
        os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = "0"
        try:
            with tempfile.TemporaryDirectory() as td:
                first = run_tiny_real_data_validation(
                    token_cache_dir=token_dir,
                    checkpoint_dir=os.path.join(td, "first"),
                    seq_len=8,
                    seed=777,
                    deterministic=True,
                )
                second = run_tiny_real_data_validation(
                    token_cache_dir=token_dir,
                    checkpoint_dir=os.path.join(td, "second"),
                    seq_len=8,
                    seed=777,
                    deterministic=True,
                )

            self.assertEqual(first["artifact_path"], second["artifact_path"])
            self.assertEqual(first["tokens_read"], second["tokens_read"])
            self.assertEqual(first["model_config_hash"], second["model_config_hash"])
            self.assertEqual(first["loss"], second["loss"])
        finally:
            if old_manifest_policy is None:
                os.environ.pop("MYLLM_WRITE_CHECKPOINT_MANIFEST", None)
            else:
                os.environ["MYLLM_WRITE_CHECKPOINT_MANIFEST"] = old_manifest_policy


if __name__ == "__main__":
    unittest.main()
