import json
import unittest

from eval_harness.hidden_eval_assets import (
    CATEGORY_NAMES,
    HOLDOUT_REGISTRY_PATH,
    PRIVATE_TARGETS_PATH,
    SEED_SET_PATH,
    SPEC_JSON_PATH,
    SPEC_MD_PATH,
    load_hidden_eval_holdout_registry,
    load_hidden_eval_private_targets,
    load_hidden_eval_seed_set,
    load_hidden_eval_spec,
    validate_hidden_eval_assets,
)


class HiddenEvalAssetTests(unittest.TestCase):
    def test_hidden_eval_artifacts_exist_and_validate(self):
        report = validate_hidden_eval_assets()
        self.assertEqual(report["schema"], "hidden_eval_asset_validation_v1")
        self.assertEqual(report["item_count"], 9)
        self.assertGreater(report["split_counts"]["frozen"], 0)
        self.assertGreater(report["split_counts"]["rotating"], 0)

    def test_seed_set_covers_all_categories_with_unique_ids(self):
        items = load_hidden_eval_seed_set()
        self.assertEqual(len({item["id"] for item in items}), len(items))
        self.assertEqual({item["category"] for item in items}, set(CATEGORY_NAMES))

    def test_registry_matches_seed_item_count_and_split_summary(self):
        items = load_hidden_eval_seed_set()
        registry = load_hidden_eval_holdout_registry()
        self.assertEqual(registry["schema"], "hidden_eval_holdout_hash_registry_v1")
        self.assertEqual(len(registry["items"]), len(items))
        self.assertGreater(registry["split_summary"]["frozen"], 0)
        self.assertGreater(registry["split_summary"]["rotating"], 0)

    def test_private_targets_cover_all_seed_items(self):
        items = load_hidden_eval_seed_set()
        private_targets = load_hidden_eval_private_targets()
        for item in items:
            self.assertIn(item["private_target_ref"], private_targets)

    def test_spec_and_artifacts_do_not_claim_public_benchmark_scores(self):
        texts = [
            SPEC_MD_PATH.read_text(encoding="utf-8"),
            SPEC_JSON_PATH.read_text(encoding="utf-8"),
            SEED_SET_PATH.read_text(encoding="utf-8"),
            HOLDOUT_REGISTRY_PATH.read_text(encoding="utf-8"),
            PRIVATE_TARGETS_PATH.read_text(encoding="utf-8"),
        ]
        combined = "\n".join(texts).lower()
        self.assertNotIn("leaderboard", combined)
        self.assertNotIn("public benchmark score", combined)
        self.assertNotIn("1x/10x/100x", combined)

    def test_machine_readable_spec_has_required_categories(self):
        spec = load_hidden_eval_spec()
        self.assertEqual(spec["schema"], "hidden_eval_spec_v1")
        self.assertEqual(set(spec["categories"].keys()), set(CATEGORY_NAMES))
        self.assertIn("scoring_rules", spec)
        self.assertGreater(len(spec["scoring_rules"]), 0)


if __name__ == "__main__":
    unittest.main()
