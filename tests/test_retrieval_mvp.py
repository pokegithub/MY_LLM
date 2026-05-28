import json
import os
import tempfile
import unittest

from eval_harness.hidden_seed_runner import run_hidden_eval_seed_execution
from retrieval.local_repo import (
    ANSWER_SCHEMA,
    abstain_for_missing_evidence,
    assemble_cited_answer,
    build_index_for_paths,
    build_local_repo_docs_index,
    citation_check_for_query,
    extract_answer_claims,
    search_index,
    validate_cited_answer,
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

    def test_citation_validator_rejects_invalid_locator_and_hidden_private_path(self):
        with tempfile.TemporaryDirectory() as td:
            private_dir = os.path.join(td, "evals", "hidden", "private")
            os.makedirs(private_dir)
            private_path = os.path.join(private_dir, "secret.md")
            with open(private_path, "w", encoding="utf-8") as handle:
                handle.write("secret private target text\n")
            index = build_index_for_paths([private_path], workspace_root=td)
            citation = search_index("secret", index=index)["citations"][0]

        self.assertFalse(validate_citation(citation, index)["valid"])
        citation = dict(citation)
        citation["document_ref"] = "SYSTEM_MAP.md"
        citation["locator"] = "999-1000"
        self.assertFalse(validate_citation(citation, index)["valid"])

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

    def test_cited_answer_surface_requires_valid_citations(self):
        with tempfile.TemporaryDirectory() as td:
            index = build_local_repo_docs_index(output_path=os.path.join(td, "index.json"))
            answer = assemble_cited_answer("What is the backend failure contract?", index=index)

        self.assertEqual(answer["schema"], ANSWER_SCHEMA)
        self.assertEqual(answer["answer_status"], "answered_with_citations")
        self.assertTrue(answer["citations"])
        self.assertIn("[c", answer["answer_text"])
        validation = validate_cited_answer(answer, index)
        self.assertTrue(validation["valid"])
        self.assertTrue(validation["answer_verifier_valid"])
        self.assertGreater(validation["claim_count"], 0)
        self.assertEqual(validation["claims_unsupported"], 0)
        self.assertIn(validation["claim_support_level"], {"direct_quote_or_near_quote", "lexical_overlap"})
        self.assertEqual(validation["semantic_truth_claim"], "limited_or_none")

    def test_cited_answer_verifier_rejects_missing_citations_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            index = build_local_repo_docs_index(output_path=os.path.join(td, "index.json"))
            answer = assemble_cited_answer("backend failure contract", index=index)

        no_citations = dict(answer)
        no_citations["citations"] = []
        self.assertFalse(validate_cited_answer(no_citations, index)["valid"])
        bad_hash = dict(answer)
        bad_hash["citations"] = [dict(answer["citations"][0], support_snippet_hash="bad")]
        self.assertFalse(validate_cited_answer(bad_hash, index)["valid"])

    def test_claim_extraction_and_verifier_catch_uncited_or_unsupported_claims(self):
        with tempfile.TemporaryDirectory() as td:
            index = build_local_repo_docs_index(output_path=os.path.join(td, "index.json"))
            answer = assemble_cited_answer("backend failure contract", index=index)

        claims = extract_answer_claims(answer["answer_text"])
        self.assertTrue(claims)
        self.assertTrue(all("claim_id" in claim for claim in claims))

        uncited = dict(answer)
        uncited["answer_text"] = answer["answer_text"] + " This sentence has no citation."
        uncited_validation = validate_cited_answer(uncited, index)
        self.assertFalse(uncited_validation["valid"])
        self.assertIn("unsupported_claims_present", uncited_validation["failure_reasons"])
        self.assertGreater(uncited_validation["claims_unsupported"], 0)

        unsupported = dict(answer)
        unsupported["answer_text"] = answer["answer_text"] + " The backend is production ready. [c1]"
        unsupported_validation = validate_cited_answer(unsupported, index)
        self.assertFalse(unsupported_validation["valid"])
        self.assertGreater(unsupported_validation["claims_unsupported"], 0)
        self.assertTrue(
            any(
                claim.get("unsupported_reason") == "weak_lexical_support"
                for claim in unsupported_validation["unsupported_claims"]
            )
        )

    def test_answer_surface_abstains_for_out_of_scope_and_does_not_invent(self):
        with tempfile.TemporaryDirectory() as td:
            index = build_local_repo_docs_index(output_path=os.path.join(td, "index.json"))
            out_of_scope = assemble_cited_answer("What happened in today's world news?", index=index)
            weak_evidence = assemble_cited_answer("qzzx backend", index=index)
            no_evidence = assemble_cited_answer("qzzx nonexistent private miracle claim", index=index)

        self.assertEqual(out_of_scope["answer_status"], "unsupported_current_phase")
        self.assertEqual(out_of_scope["citations"], [])
        self.assertEqual(weak_evidence["answer_status"], "abstained_no_evidence")
        self.assertEqual(weak_evidence["abstention_reason"], "weak_lexical_evidence")
        self.assertEqual(no_evidence["answer_status"], "abstained_no_evidence")
        self.assertEqual(no_evidence["citations"], [])

    def test_retrieval_answer_module_has_no_web_vector_or_memory_path(self):
        with open(os.path.join("retrieval", "local_repo.py"), "r", encoding="utf-8") as handle:
            source = handle.read().lower()
        self.assertNotIn("requests.", source)
        self.assertNotIn("urllib.request", source)
        self.assertNotIn("faiss", source)
        self.assertNotIn("chromadb", source)
        self.assertNotIn("embedding", source)

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
        self.assertEqual(report["counts"]["answer_verifier_passed"], 2)
        self.assertEqual(report["counts"]["claim_verifier_passed"], 2)
        self.assertEqual(report["counts"]["claims_unsupported"], 0)
        for item in report["items"]:
            self.assertGreater(item["claim_count"], 0)
            self.assertEqual(item["claims_unsupported"], 0)
            self.assertTrue(item["claim_verifier_passed"])
        self.assertFalse(report["private_target_exposed_in_public_summary"])
        self.assertFalse(report["answer_exposed_in_public_summary"])
        self.assertNotIn("required_claim_fragments", public_summary)
        self.assertNotIn("expected_output", public_summary)


if __name__ == "__main__":
    unittest.main()
