import json
import os
import unittest


class BakeoffProtocolAssetTests(unittest.TestCase):
    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_bakeoff_protocol_artifacts_exist(self):
        self.assertTrue(os.path.isfile("./model_selection/bakeoff_protocol_v1.md"))
        self.assertTrue(os.path.isfile("./model_selection/bakeoff_protocol_v1.json"))
        self.assertTrue(os.path.isfile("./model_selection/bakeoff_candidate_matrix_v1.json"))
        self.assertTrue(os.path.isfile("./model_selection/bakeoff_scoring_v1.md"))
        self.assertTrue(os.path.isfile("./model_selection/bakeoff_scoring_v1.json"))

    def test_candidate_matrix_has_exactly_three_dense_only_slots(self):
        payload = self._load_json("./model_selection/bakeoff_candidate_matrix_v1.json")

        self.assertEqual(payload["schema"], "bakeoff_candidate_matrix_v1")
        self.assertEqual(payload["candidate_slot_count"], 3)
        self.assertTrue(payload["dense_only_required"])
        self.assertFalse(payload["moe_allowed"])
        self.assertEqual(len(payload["candidate_slots"]), 3)
        self.assertEqual(
            [item["slot_id"] for item in payload["candidate_slots"]],
            ["current_custom_base", "open_dense_7b_slot", "open_dense_14b_slot"],
        )
        self.assertEqual(
            len({item["candidate_id"] for item in payload["candidate_slots"]}),
            3,
        )
        self.assertTrue(
            all(item["dense_or_moe"] != "moe" for item in payload["candidate_slots"])
        )

    def test_custom_base_slot_requires_real_checkpoint(self):
        payload = self._load_json("./model_selection/bakeoff_candidate_matrix_v1.json")
        custom = payload["candidate_slots"][0]

        self.assertEqual(custom["slot_id"], "current_custom_base")
        self.assertEqual(custom["custom_base_qualification_status"], "requires_real_checkpoint")
        self.assertIn(
            "no_usable_checkpoint_evidenced_in_standard_checkpoint_surface",
            custom["disqualifier_flags"],
        )
        self.assertIn("not a full contender", custom["notes"].lower())

    def test_scoring_rubric_has_required_dimensions_and_weights(self):
        payload = self._load_json("./model_selection/bakeoff_scoring_v1.json")
        expected = {
            "hidden_exact_symbolic_correctness",
            "hidden_coding_final_verified_success",
            "hidden_coding_repair_loop_success",
            "abstention_precision",
            "source_grounded_truthfulness",
            "retrieval_grounded_qa",
            "runtime_fit_stability",
            "structured_output_reliability",
            "cost_latency_per_verified_success",
            "deployment_fit_linux_first_workstation",
        }

        self.assertEqual(payload["schema"], "bakeoff_scoring_v1")
        self.assertEqual(sum(item["weight"] for item in payload["dimensions"]), 100)
        self.assertEqual(
            {item["dimension_id"] for item in payload["dimensions"]},
            expected,
        )

    def test_protocol_preserves_evidence_first_truthfulness(self):
        with open("./model_selection/bakeoff_protocol_v1.md", "r", encoding="utf-8") as handle:
            protocol_text = handle.read().lower()
        with open("./model_selection/bakeoff_scoring_v1.md", "r", encoding="utf-8") as handle:
            scoring_text = handle.read().lower()
        with open("./model_selection/bakeoff_protocol_v1.json", "r", encoding="utf-8") as handle:
            protocol_json_text = handle.read().lower()

        combined = "\n".join([protocol_text, scoring_text, protocol_json_text])
        self.assertIn("public benchmark prestige", combined)
        self.assertIn("hidden private evals", combined)
        self.assertTrue(
            "no candidate is selected by popularity or sentiment" in combined
            or "no_candidate_is_preselected" in combined
        )
        self.assertIn("no moe candidates", combined)
        self.assertNotIn("winner already", combined)
        self.assertNotIn("best model by default", combined)


if __name__ == "__main__":
    unittest.main()
