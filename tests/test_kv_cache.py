import unittest

import torch

from config import EDGE_0_5B_GQA, build_model_config
from model import LLM


class KVCacheTests(unittest.TestCase):
    def test_local_attention_kv_cache_is_bounded(self):
        torch.manual_seed(4)
        cfg = build_model_config(
            EDGE_0_5B_GQA,
            vocab_size=128,
            dim=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            ffn_dim=128,
            max_seq_len=32,
            sliding_window=3,
        )
        model = LLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))

        output = model(ids)
        kv_cache = output.kv_cache

        local_layer_count = cfg.n_layers // 2
        for layer_idx in range(local_layer_count):
            self.assertLessEqual(
                kv_cache[layer_idx][0].shape[2],
                cfg.sliding_window,
            )
            self.assertLessEqual(
                kv_cache[layer_idx][1].shape[2],
                cfg.sliding_window,
            )
        for layer_idx in range(local_layer_count, cfg.n_layers):
            self.assertEqual(kv_cache[layer_idx][0].shape[2], ids.shape[1])

        offset = ids.shape[1]
        next_id = torch.randint(0, cfg.vocab_size, (1, 1))
        for _ in range(6):
            output = model(next_id, kv_cache=kv_cache, offset=offset)
            kv_cache = output.kv_cache
            for layer_idx in range(local_layer_count):
                self.assertLessEqual(
                    kv_cache[layer_idx][0].shape[2],
                    cfg.sliding_window,
                )
                self.assertLessEqual(
                    kv_cache[layer_idx][1].shape[2],
                    cfg.sliding_window,
                )
            offset += 1
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)


if __name__ == "__main__":
    unittest.main()
