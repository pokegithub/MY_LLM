import unittest

import torch

from config import EDGE_0_5B_GQA, build_model_config
from model import LLM, LossOutput, ModelOutput


def tiny_production_config(**overrides):
    values = {
        "vocab_size": 128,
        "dim": 64,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 128,
        "max_seq_len": 32,
        "sliding_window": 4,
    }
    values.update(overrides)
    return build_model_config(EDGE_0_5B_GQA, **values)


class ModelShapeTests(unittest.TestCase):
    def test_production_model_has_no_experimental_modules(self):
        cfg = tiny_production_config()
        model = LLM(cfg)

        self.assertIsNone(model.memory)
        self.assertIsNone(model.conf_head)
        self.assertEqual(model._n_moe_layers, 0)
        self.assertTrue(all(not block.is_moe for block in model.layers))
        self.assertTrue(all(not block.use_mod for block in model.layers))

    def test_forward_eval_shapes(self):
        torch.manual_seed(1)
        cfg = tiny_production_config()
        model = LLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 7))

        output = model(ids)

        self.assertIsInstance(output, ModelOutput)
        self.assertEqual(tuple(output.logits.shape), (2, 7, cfg.vocab_size))
        self.assertIsNone(output.unc_scores)
        self.assertEqual(len(output.kv_cache), cfg.n_layers)
        for kv in output.kv_cache:
            self.assertIsNotNone(kv)
            self.assertEqual(tuple(kv[0].shape[:2]), (2, cfg.n_kv_heads))
            self.assertEqual(kv[0].shape[-1], cfg.head_dim)

    def test_forward_training_loss_shapes(self):
        torch.manual_seed(2)
        cfg = tiny_production_config()
        model = LLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 6))
        targets = torch.randint(0, cfg.vocab_size, (2, 6))

        output = model(ids, targets=targets)

        self.assertIsInstance(output, LossOutput)
        self.assertEqual(tuple(output.total_loss.shape), ())
        self.assertEqual(tuple(output.ce_loss.shape), ())
        self.assertEqual(output.unc_loss.item(), 0.0)
        self.assertEqual(output.aux_loss.item(), 0.0)

    def test_generate_smoke(self):
        torch.manual_seed(3)
        cfg = tiny_production_config()
        model = LLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 5))

        generated, uncertainty = model.generate(
            ids,
            max_new_tokens=2,
            temperature=0.0,
            eos_token_id=-1,
        )

        self.assertEqual(tuple(generated.shape), (1, 7))
        self.assertIsNone(uncertainty)


if __name__ == "__main__":
    unittest.main()
