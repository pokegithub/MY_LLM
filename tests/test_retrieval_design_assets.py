import json
import os
import unittest


class RetrievalDesignAssetTests(unittest.TestCase):
    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_retrieval_design_artifacts_exist(self):
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_design_v1.md"))
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_design_v1.json"))
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_source_policy_v1.md"))
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_source_policy_v1.json"))
        self.assertTrue(os.path.isfile("./retrieval_design/citation_contract_v1.md"))
        self.assertTrue(os.path.isfile("./retrieval_design/citation_contract_v1.json"))
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_abstention_policy_v1.md"))
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_verifier_contract_v1.md"))
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_verifier_contract_v1.json"))
        self.assertTrue(os.path.isfile("./retrieval_design/retrieval_benchmark_exclusion_policy_v1.md"))

    def test_exactly_one_narrow_first_scope_is_defined(self):
        payload = self._load_json("./retrieval_design/retrieval_design_v1.json")
        scope = payload["primary_scope"]

        self.assertEqual(payload["schema"], "retrieval_design_v1")
        self.assertEqual(scope["scope_count"], 1)
        self.assertEqual(scope["scope_id"], "repo_local_docs_and_manifests_only")
        self.assertEqual(scope["status"], "planned_not_implemented")
        self.assertIn("repo_authored_local_docs", scope["allowed_document_classes"])
        self.assertIn("repo_generated_local_manifests", scope["allowed_document_classes"])
        self.assertIn("vector_or_semantic_retrieval", scope["non_goals"])

    def test_source_policy_classes_are_complete_and_conservative(self):
        payload = self._load_json("./retrieval_design/retrieval_source_policy_v1.json")
        classes = {item["class_id"]: item for item in payload["source_classes"]}

        self.assertEqual(payload["schema"], "retrieval_source_policy_v1")
        self.assertEqual(payload["legal_clearance_claim"], "none")
        self.assertEqual(set(classes.keys()), {"allowed", "restricted", "blocked", "benchmark_excluded", "unknown"})
        self.assertTrue(classes["allowed"]["retrieval_allowed"])
        self.assertFalse(classes["restricted"]["retrieval_allowed"])
        self.assertFalse(classes["blocked"]["retrieval_allowed"])
        self.assertFalse(classes["benchmark_excluded"]["retrieval_allowed"])
        self.assertFalse(classes["unknown"]["retrieval_allowed"])

    def test_citation_and_verifier_contracts_have_required_fields(self):
        citation = self._load_json("./retrieval_design/citation_contract_v1.json")
        verifier = self._load_json("./retrieval_design/retrieval_verifier_contract_v1.json")

        self.assertEqual(citation["schema"], "citation_contract_v1")
        self.assertIn("citation_id", citation["required_fields"])
        self.assertIn("source_id", citation["required_fields"])
        self.assertIn("document_ref", citation["required_fields"])
        self.assertIn("locator", citation["required_fields"])
        self.assertIn("support_snippet_hash", citation["required_fields"])
        self.assertEqual(verifier["schema"], "retrieval_verifier_contract_v1")
        self.assertEqual(verifier["verifier_authority"], "required")
        self.assertIn(
            "citation_presence_for_source_grounded_answers",
            {item["check_id"] for item in verifier["checks"]},
        )
        self.assertIn(
            "benchmark_exclusion_compliance",
            {item["check_id"] for item in verifier["checks"]},
        )

    def test_artifacts_preserve_truthful_nonimplementation_language(self):
        paths = [
            "./retrieval_design/retrieval_design_v1.md",
            "./retrieval_design/retrieval_design_v1.json",
            "./retrieval_design/retrieval_source_policy_v1.md",
            "./retrieval_design/citation_contract_v1.md",
            "./retrieval_design/retrieval_abstention_policy_v1.md",
            "./retrieval_design/retrieval_verifier_contract_v1.md",
            "./retrieval_design/retrieval_benchmark_exclusion_policy_v1.md",
        ]
        texts = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                texts.append(handle.read().lower())
        combined = "\n".join(texts)

        self.assertIn("planned", combined)
        self.assertIn("not implemented", combined)
        self.assertIn("verifier", combined)
        self.assertIn("benchmark-excluded", combined)
        self.assertIn("not proof", combined)
        self.assertNotIn("already integrated", combined)
        self.assertNotIn("production ready", combined)
        self.assertNotIn("web search is included", combined)
        self.assertNotIn("vector search is included", combined)


if __name__ == "__main__":
    unittest.main()
