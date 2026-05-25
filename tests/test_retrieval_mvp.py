import json
import os
import tempfile
import unittest

from eval_harness.hidden_seed_runner import run_hidden_eval_seed_execution
from retrieval.local_repo import (
    abstain_for_missing_evidence,
    build_local_repo_docs_index,
    citation_check_for_query,
    search_index,
    validate_citation,
    validate_citation_bundle,
)


class RetrievalMvpTests(unittest.TestCase):
    def test_index_build_includes_allowed_docs_and_excludes_private_material(self):
        with tempfile.TemporaryDirectory() as td:
            index_path = os.path.join(td, "index.json")
            index = build_local_repo_docs_index(output_path=index_path)

        refs = {doc["document_ref"] for doc in index["documents"]}
        self.assertIn("SYSTEM_MAP.md", refs)
        self.assertIn("retrieval_design/citation_contract_v1.md", refs)
        self.assertIn("evals/hidden/hidden_eval_spec_v1.md", refs)
        self.assertFalse(any(ref.startswith("evals/hidden/private") for ref in refs))
        self.assertFalse(any(ref.startswith("evals/hidden/fixtures") for ref in refs))
        self.assertFalse(any(ref.startswith("run_artifacts/local_models") for ref in refs))
        self.assertFalse(any(ref.startswith("data_cache") for ref in refs))

    def test_search_returns_checkable_citations(self):
        with tempfile.TemporaryDirectory() as td:
            index_path = os.path.join(td, "index.json")
            index = build_local_repo_docs_index(output_path=index_path)
            result = search_index("backend failure contract", index=index)

        self.assertEqual(result["status"], "evidence_found")
        self.assertTrue(result["citations"])
        citation = result["citations"][0]
        for field in (
            "citation_id",
            "source_class",
            "source_id",
            "document_ref",
            "locator_type",
            "locator",
            "support_kind",
            "support_snippet_hash",
        ):
            self.assertIn(field, citation)
        validation = validate_citation(citation, index)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["locator_resolution_status"], "resolved")

    def test_citation_validator_rejects_invalid_hash_and_missing_citations(self):
        with tempfile.TemporaryDirectory() as td:
            index = build_local_repo_docs_index(output_path=os.path.join(td, "index.json"))
            result = search_index("citation required fields", index=index)

        citation = dict(result["citations"][0])
        citation["support_snippet_hash"] = "bad"
        self.assertFalse(validate_citation(citation, index)["valid"])
        self.assertFalse(validate_citation_bundle([], index)["valid"])

    def test_empty_or_out_of_scope_retrieval_abstains(self):
        self.assertEqual(
            abstain_for_missing_evidence("external current weather", reason="out_of_scope")["status"],
            "abstain",
        )
        with tempfile.TemporaryDirectory() as td:
            index = build_local_repo_docs_index(output_path=os.path.join(td, "index.json"))
            result = search_index("", index=index)
        self.assertEqual(result["status"], "abstain_no_query_terms")
        self.assertEqual(result["abstention"]["status"], "abstain")

    def test_citation_check_for_query_validates_search_citations(self):
        with tempfile.TemporaryDirectory() as td:
            index_path = os.path.join(td, "index.json")
            build_local_repo_docs_index(output_path=index_path)
            report = citation_check_for_query("backend failure contract", index_path=index_path)

        self.assertEqual(report["schema"], "retrieval_citation_check_report_v1")
        self.assertEqual(report["search_status"], "evidence_found")
        self.assertTrue(report["citation_validation"]["valid"])
        self.assertEqual(report["quality_claim"], "none")

    def test_hidden_eval_source_items_run_without_public_private_answer_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            report = run_hidden_eval_seed_execution(
                seed_ids=("hidden_truth_001", "hidden_retrieval_001"),
                workspace_root=".",
                backend_config_path=os.path.join(td, "missing_qwen_config.json"),
                output_root=os.path.join(td, "hidden_eval_runs"),
                use_default_agent_report_dir=False,
            )

        public_summary = json.dumps(report, sort_keys=True)
        self.assertEqual(report["counts"]["retrieval_routed"], 2)
        self.assertEqual(report["counts"]["retrieval_supported"], 2)
        self.assertEqual(report["counts"]["verifier_passed"], 2)
        self.assertFalse(report["private_target_exposed_in_public_summary"])
        self.assertFalse(report["answer_exposed_in_public_summary"])
        self.assertNotIn("required_claim_fragments", public_summary)
        self.assertNotIn("expected_output", public_summary)


if __name__ == "__main__":
    unittest.main()
