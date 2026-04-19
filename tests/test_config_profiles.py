import unittest

from config import (
    EDGE_0_5B_GQA,
    LEGACY_EXPERIMENTAL_439M,
    LOCAL_1_5B_GQA,
    WORKSTATION_3B_GQA,
    _restore_runtime_state,
    apply_overrides,
    available_model_profiles,
    build_model_config,
    model_cfg,
    runtime_config_dict,
)


class ConfigProfileTests(unittest.TestCase):
    def test_required_profiles_are_registered(self):
        self.assertEqual(
            set(available_model_profiles()),
            {
                EDGE_0_5B_GQA,
                LOCAL_1_5B_GQA,
                WORKSTATION_3B_GQA,
                LEGACY_EXPERIMENTAL_439M,
            },
        )

    def test_default_profile_is_dense_gqa_production_path(self):
        self.assertEqual(model_cfg.profile_name, EDGE_0_5B_GQA)
        self.assertEqual(model_cfg.enabled_experimental_features(), ())
        self.assertFalse(model_cfg.use_moe)
        self.assertEqual(model_cfg.n_memory_tokens, 0)
        self.assertEqual(model_cfg.n_loops, 1)
        self.assertFalse(model_cfg.use_confidence_head)
        self.assertFalse(model_cfg.use_mod_routing)
        self.assertGreater(model_cfg.n_heads, model_cfg.n_kv_heads)

    def test_production_profiles_are_dense_gqa_without_experimental_features(self):
        for profile_name in (EDGE_0_5B_GQA, LOCAL_1_5B_GQA, WORKSTATION_3B_GQA):
            cfg = build_model_config(profile_name)
            self.assertEqual(cfg.enabled_experimental_features(), ())
            self.assertFalse(cfg.use_moe)
            self.assertEqual(cfg.n_memory_tokens, 0)
            self.assertEqual(cfg.n_loops, 1)
            self.assertFalse(cfg.use_confidence_head)
            self.assertGreater(cfg.n_heads, cfg.n_kv_heads)
            self.assertEqual(cfg.n_heads % cfg.n_kv_heads, 0)

    def test_experimental_features_require_legacy_profile(self):
        with self.assertRaisesRegex(ValueError, "Experimental mechanisms"):
            build_model_config(EDGE_0_5B_GQA, use_moe=True)

    def test_legacy_profile_is_explicit_and_experimental(self):
        cfg = build_model_config(LEGACY_EXPERIMENTAL_439M)
        self.assertTrue(cfg.is_legacy_experimental)
        self.assertIn("use_moe", cfg.enabled_experimental_features())
        self.assertIn("n_memory_tokens", cfg.enabled_experimental_features())
        self.assertIn("use_confidence_head", cfg.enabled_experimental_features())

    def test_runtime_profile_override_applies_profile_values(self):
        snapshot = runtime_config_dict()
        try:
            apply_overrides({"model": {"profile_name": LEGACY_EXPERIMENTAL_439M}})
            self.assertEqual(model_cfg.profile_name, LEGACY_EXPERIMENTAL_439M)
            self.assertEqual(model_cfg.dim, 1024)
            self.assertTrue(model_cfg.use_moe)
            self.assertEqual(model_cfg.n_memory_tokens, 64)
            self.assertTrue(model_cfg.use_confidence_head)
        finally:
            _restore_runtime_state(snapshot)

    def test_runtime_profile_override_keeps_explicit_field_overrides(self):
        snapshot = runtime_config_dict()
        try:
            apply_overrides(
                {
                    "model": {
                        "profile_name": LOCAL_1_5B_GQA,
                        "dim": 128,
                        "n_heads": 4,
                        "n_kv_heads": 2,
                        "ffn_dim": 256,
                    }
                }
            )
            self.assertEqual(model_cfg.profile_name, LOCAL_1_5B_GQA)
            self.assertEqual(model_cfg.dim, 128)
            self.assertEqual(model_cfg.n_heads, 4)
            self.assertEqual(model_cfg.n_kv_heads, 2)
            self.assertEqual(model_cfg.ffn_dim, 256)
            self.assertFalse(model_cfg.use_moe)
        finally:
            _restore_runtime_state(snapshot)


if __name__ == "__main__":
    unittest.main()
