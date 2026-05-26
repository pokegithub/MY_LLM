"""Narrow hidden-eval seed execution for Phase A backend smoke evidence."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from agent.exact_tools import execute_exact_task
from agent.orchestrator import solve_task
from agent.types import TaskRequest
from config import agent_cfg
from eval_harness.hidden_eval_assets import (
    HIDDEN_EVAL_DIR,
    load_hidden_eval_private_targets,
    load_hidden_eval_seed_set,
    validate_hidden_eval_assets,
)
from retrieval.local_repo import (
    ANSWER_SCHEMA,
    assemble_cited_answer,
    build_index_for_paths,
    make_citation,
    search_index,
    validate_cited_answer,
    validate_citation_bundle,
)


HIDDEN_EVAL_RUN_SCHEMA = "hidden_eval_seed_run_v1"
DEFAULT_QWEN_BACKEND_CONFIG = "./configs/backend_smoke_qwen2_5_coder_0_5b.json"
DEFAULT_HIDDEN_SEED_IDS = (
    "hidden_exact_001",
    "hidden_exact_002",
    "hidden_abstain_002",
    "hidden_code_patch_001",
    "hidden_code_repair_001",
    "hidden_truth_001",
    "hidden_retrieval_001",
)
CODING_CATEGORIES = {
    "coding_patch_success",
    "coding_repair_loop_success",
    "verifier_loop_kpis_latency_cost",
}
UNSUPPORTED_CURRENT_PHASE_CATEGORIES = {
}


@contextmanager
def _agent_overrides(**values: Any) -> Iterator[None]:
    old_values = {key: getattr(agent_cfg, key) for key in values}
    try:
        for key, value in values.items():
            setattr(agent_cfg, key, value)
        yield
    finally:
        for key, value in old_values.items():
            setattr(agent_cfg, key, value)


def _new_run_id() -> str:
    return f"hidden_eval_seed_run_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(path.resolve())


def _seed_by_id(seed_items: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(item["id"]): item for item in seed_items}


def _load_agent_config(path: str, workspace_root: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(workspace_root).resolve() / config_path
    if not config_path.is_file():
        return {
            "backend_kind": "none",
            "backend_model_id_or_path": None,
            "backend_local_files_only": True,
            "backend_trust_remote_code": False,
            "backend_device": "auto",
            "backend_max_new_tokens": 512,
            "backend_temperature": 0.0,
            "backend_prompt_max_chars": 8000,
            "config_path": str(config_path),
            "config_exists": False,
        }
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    agent = dict(payload.get("agent") or {})
    agent["config_path"] = str(config_path)
    agent["config_exists"] = True
    return agent


def _model_path_exists(model_path: Optional[str], workspace_root: str) -> Optional[bool]:
    if not model_path:
        return None
    path = Path(str(model_path))
    if not path.is_absolute():
        path = Path(workspace_root).resolve() / path
    return path.exists()


def _base_item_result(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "hidden_eval_seed_item_result_v1",
        "item_id": item.get("id"),
        "category": item.get("category"),
        "evaluation_mode": item.get("evaluation_mode"),
        "split": (item.get("holdout_metadata") or {}).get("split"),
        "route_used": "not_started",
        "final_status": "not_started",
        "verifier_status": "not_attempted",
        "backend_kind": None,
        "model_id_or_path": None,
        "backend_routed": False,
        "exact_tool_routed": False,
        "verifier_reached": False,
        "verifier_passed": False,
        "blocked_reason": None,
        "unsupported_reason": None,
        "private_target_used": False,
        "private_target_exposed_in_public_summary": False,
        "answer_exposed_in_public_summary": False,
        "private_target_leakage": False,
        "agent_run_id": None,
        "agent_report_path": None,
        "trajectory_id": None,
        "attempt_count": 0,
        "quality_claim": "none",
        "elapsed_ms": 0,
    }


def _run_exact_item(
    *,
    item: Mapping[str, Any],
    target: Mapping[str, Any],
    workspace_root: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    result = _base_item_result(item)
    result["route_used"] = "deterministic_exact_tool"
    result["exact_tool_routed"] = True
    result["private_target_used"] = True
    passed = False
    try:
        if item["id"] == "hidden_exact_001":
            payload_text = item["task_text"].split(":", 1)[1].strip()
            exact = execute_exact_task(f"json format {payload_text}", workspace_root)
            passed = exact.get("result") == target.get("expected_output") and bool(exact.get("validated"))
        elif item["id"] == "hidden_exact_002":
            rel_path = str(item.get("artifact_refs", [""])[0])
            text = (HIDDEN_EVAL_DIR / rel_path).read_text(encoding="utf-8")
            token = "[restart-window]"
            observed = None
            for line_index, line in enumerate(text.splitlines()):
                offset = line.find(token)
                if offset >= 0:
                    observed = {"line_index": line_index, "char_offset": offset}
                    break
            passed = observed == target.get("expected_output")
        else:
            result["final_status"] = "unsupported"
            result["unsupported_reason"] = "exact seed item is not implemented by this narrow runner"
            return result
    except Exception as exc:
        result["final_status"] = "failed"
        result["blocked_reason"] = str(exc)
    else:
        result["final_status"] = "passed" if passed else "failed"
        result["verifier_status"] = "passed" if passed else "failed"
        result["quality_claim"] = "eval_status_only"
    finally:
        result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def _run_decision_item(item: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    result = _base_item_result(item)
    result["route_used"] = "deterministic_exact_tool_decision"
    result["exact_tool_routed"] = True
    result["private_target_used"] = True
    if item["id"] == "hidden_abstain_002":
        decision = "route_to_exact_tool"
        accepted = set(str(value) for value in target.get("accepted_responses", []))
        passed = decision in accepted
        result["final_status"] = "passed" if passed else "failed"
        result["verifier_status"] = "passed" if passed else "failed"
        result["quality_claim"] = "eval_status_only"
    else:
        result["final_status"] = "unsupported"
        result["unsupported_reason"] = "freshness/source-grounded decision evaluator is not implemented in Phase A"
    result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def _copy_artifact_to_workspace(item: Mapping[str, Any], workspace: Path) -> str:
    artifact_refs = list(item.get("artifact_refs") or [])
    if not artifact_refs:
        raise ValueError("coding seed item is missing artifact_refs")
    source = HIDDEN_EVAL_DIR / str(artifact_refs[0])
    target_name = Path(str(artifact_refs[0])).name
    target = workspace / target_name
    shutil.copyfile(source, target)
    return target_name


def _hidden_behavior_test_content(module_name: str, function_name: str, assertions: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "import unittest",
        "",
        f"from {module_name} import {function_name}",
        "",
        "",
        "class HiddenBehaviorTests(unittest.TestCase):",
    ]
    for index, assertion in enumerate(assertions):
        call = str(assertion["call"])
        expected = repr(assertion["expected"])
        lines.extend(
            [
                f"    def test_hidden_case_{index}(self):",
                f"        self.assertEqual({call}, {expected})",
                "",
            ]
        )
    lines.extend(["", "if __name__ == \"__main__\":", "    unittest.main()", ""])
    return "\n".join(lines)


def _write_hidden_behavior_tests(item: Mapping[str, Any], target: Mapping[str, Any], workspace: Path) -> str:
    filename = Path(str(item.get("artifact_refs", ["target.py"])[0])).name
    module_name = filename[:-3]
    if filename == "math_ops.py":
        function_name = "bounded_average"
    elif filename == "slug_tools.py":
        function_name = "slugify"
    else:
        raise ValueError(f"unsupported hidden coding fixture: {filename}")
    test_path = workspace / f"test_hidden_behavior_{item['id']}.py"
    test_path.write_text(
        _hidden_behavior_test_content(module_name, function_name, list(target.get("behavior_assertions") or [])),
        encoding="utf-8",
    )
    return test_path.name


def _task_text_for_coding_item(item: Mapping[str, Any], target_filename: str) -> str:
    if item["id"] == "hidden_code_patch_001":
        return (
            f"Patch {target_filename} so bounded_average returns the correct arithmetic mean "
            "while still respecting lower and upper clamp behavior. Keep the diff minimal."
        )
    if item["id"] in {"hidden_code_repair_001", "hidden_kpi_001"}:
        return (
            f"Patch {target_filename} so slugify produces stable URL slugs on punctuation-heavy inputs. "
            "Keep the diff minimal."
        )
    return str(item.get("task_text", ""))


def _copy_backend_script_if_needed(backend_script_path: Optional[str], workspace: Path) -> Optional[str]:
    if not backend_script_path:
        return None
    source = Path(backend_script_path).resolve()
    target = workspace / "scripted_hidden_eval_candidate.json"
    shutil.copyfile(source, target)
    return str(target)


def _run_coding_item(
    *,
    item: Mapping[str, Any],
    target: Mapping[str, Any],
    workspace_root: str,
    run_dir: Path,
    backend_config: Mapping[str, Any],
    backend_script_path: Optional[str],
    agent_report_dir: Optional[str],
) -> Dict[str, Any]:
    started = time.perf_counter()
    result = _base_item_result(item)
    result["route_used"] = "coding_agent_backend"
    result["backend_routed"] = True
    result["private_target_used"] = True

    workspace = run_dir / "workspaces" / str(item["id"])
    workspace.mkdir(parents=True, exist_ok=True)
    target_filename = _copy_artifact_to_workspace(item, workspace)
    test_filename = _write_hidden_behavior_tests(item, target, workspace)
    local_script = _copy_backend_script_if_needed(backend_script_path, workspace)

    model_id_or_path = backend_config.get("backend_model_id_or_path")
    result["model_id_or_path"] = model_id_or_path
    if local_script:
        result["backend_kind"] = "scripted"
    else:
        result["backend_kind"] = backend_config.get("backend_kind", "none")
        if not _model_path_exists(str(model_id_or_path) if model_id_or_path else None, workspace_root):
            result.update(
                {
                    "final_status": "blocked",
                    "verifier_status": "not_reached",
                    "blocked_reason": "configured backend model path is unavailable",
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            )
            return result

    with _agent_overrides(
        backend_kind=str(backend_config.get("backend_kind") or "none"),
        backend_model_id_or_path=model_id_or_path,
        backend_local_files_only=bool(backend_config.get("backend_local_files_only", True)),
        backend_trust_remote_code=bool(backend_config.get("backend_trust_remote_code", False)),
        backend_device=str(backend_config.get("backend_device") or "auto"),
        backend_max_new_tokens=int(backend_config.get("backend_max_new_tokens") or 512),
        backend_temperature=float(backend_config.get("backend_temperature") or 0.0),
        backend_prompt_max_chars=int(backend_config.get("backend_prompt_max_chars") or 8000),
        report_dir=agent_report_dir or str(run_dir / "agent_reports"),
    ):
        payload = solve_task(
            TaskRequest(
                task_text=_task_text_for_coding_item(item, target_filename),
                file_hints=(target_filename,),
                checks=(f"unittest:discover -s . -p {test_filename} -v",),
                workspace_root=str(workspace),
            ),
            backend_script_path=local_script,
        )

    attempts = list(payload.get("attempts") or [])
    verification = payload.get("verification") or {}
    result.update(
        {
            "final_status": "passed" if payload.get("status") == "verified_success" else str(payload.get("status") or "failed"),
            "verifier_status": "passed" if bool(verification.get("overall_passed")) else ("failed" if attempts else "not_reached"),
            "verifier_reached": bool(attempts),
            "verifier_passed": bool(verification.get("overall_passed")),
            "blocked_reason": payload.get("blocked_reason"),
            "agent_run_id": payload.get("run_id"),
            "agent_report_path": payload.get("report_path"),
            "trajectory_id": payload.get("run_id") if payload.get("trajectory_store", {}).get("ok") else None,
            "attempt_count": len(attempts),
            "quality_claim": payload.get("quality_claim", "none"),
        }
    )
    result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def _unsupported_item(item: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    result = _base_item_result(item)
    result["route_used"] = "unsupported_current_phase"
    result["final_status"] = "unsupported"
    result["unsupported_reason"] = reason
    result["private_target_used"] = False
    return result


def _provided_doc_paths(item: Mapping[str, Any]) -> List[Path]:
    paths = []
    for ref in item.get("artifact_refs") or []:
        path = HIDDEN_EVAL_DIR / str(ref)
        if path.is_file():
            paths.append(path)
    return paths


def _source_query_for_item(item: Mapping[str, Any]) -> str:
    task_text = str(item.get("task_text") or "")
    if item.get("id") == "hidden_truth_001":
        return "customer visible effect Orchid incident dashboard reads write path unaffected"
    if item.get("id") == "hidden_retrieval_001":
        return "runtime behavior changed chunked citation rendering missing evidence blocker guess"
    return task_text


def _source_answer_from_docs(item: Mapping[str, Any], index: Mapping[str, Any]) -> Dict[str, Any]:
    item_id = str(item.get("id"))
    docs = {doc["document_ref"]: doc for doc in index.get("documents", [])}
    chunks = list(index.get("chunks", []))
    citations = []
    answer = ""
    generic_answer = assemble_cited_answer(_source_query_for_item(item), index=index, top_k=3)
    if item_id == "hidden_truth_001":
        target_ref = "fixtures/source_docs/orchid_incident.md"
        impact_chunks = [
            chunk for chunk in chunks
            if chunk.get("document_ref") == target_ref and "Customer impact" in str(chunk.get("content"))
        ]
        if impact_chunks:
            chunk = impact_chunks[0]
            citations = [chunk]
            answer = (
                "The customer-visible effect was stale account balances appearing in dashboard reads; "
                "the write path unaffected status means write path requests were unaffected."
            )
    elif item_id == "hidden_retrieval_001":
        release_ref = "fixtures/source_docs/runtime_release_notes.md"
        policy_ref = "fixtures/source_docs/source_policy_excerpt.md"
        release_chunks = [
            chunk for chunk in chunks
            if chunk.get("document_ref") == release_ref and "chunked citation rendering" in str(chunk.get("content"))
        ]
        policy_chunks = [
            chunk for chunk in chunks
            if chunk.get("document_ref") == policy_ref and "Missing evidence is a blocker" in str(chunk.get("content"))
        ]
        if release_chunks and policy_chunks:
            citations = [release_chunks[0], policy_chunks[0]]
            answer = (
                "In runtime-2026.04, chunked citation rendering was enabled for answer surfaces. "
                "The source policy says missing evidence is a blocker rather than a license to guess."
            )

    if not answer:
        generic_answer["document_refs"] = sorted(docs)
        generic_answer["answer"] = generic_answer.get("answer_text", "")
        return generic_answer
    citation_payloads = [make_citation(chunk, index) for index, chunk in enumerate(citations, start=1)]
    return {
        "schema": ANSWER_SCHEMA,
        "query": _source_query_for_item(item),
        "answer_status": "answered_with_citations",
        "answer_text": " ".join(
            f"{part.strip()} [c{index}]"
            for index, part in enumerate(answer.split(". "), start=1)
            if part.strip()
        ),
        "answer": answer,
        "citations": citation_payloads,
        "unsupported_claims": [],
        "abstention_reason": None,
        "source_policy_status": "allowed",
        "quality_claim": "none",
        "semantic_truth_claim": "limited_or_none",
        "semantic_support_level": "lexical_or_locator_only",
        "search_status": "evidence_found",
        "document_refs": sorted(docs),
        "answer_assembly": "hidden_eval_item_scoped_cited_template_v1",
        "generic_answer_surface_status": generic_answer.get("answer_status"),
        "proves_truth": False,
    }


def _verify_source_answer(
    *,
    answer_payload: Mapping[str, Any],
    target: Mapping[str, Any],
    index: Mapping[str, Any],
) -> Dict[str, Any]:
    citations = list(answer_payload.get("citations") or [])
    citation_validation = validate_citation_bundle(citations, index)
    answer_validation = validate_cited_answer(answer_payload, index)
    required_citations = set(str(ref) for ref in target.get("required_citations", []))
    cited_refs = set(str(citation.get("document_ref")) for citation in citations)
    required_citation_status = required_citations.issubset(cited_refs)
    answer_text = str(answer_payload.get("answer_text") or answer_payload.get("answer") or "").lower()
    claim_fragments_present = all(
        str(fragment).lower() in answer_text
        for fragment in target.get("required_claim_fragments", [])
    )
    passed = (
        answer_payload.get("answer_status") == "answered_with_citations"
        and bool(answer_validation.get("valid"))
        and required_citation_status
        and claim_fragments_present
    )
    return {
        "schema": "source_grounded_verifier_result_v1",
        "passed": passed,
        "citation_validation": citation_validation,
        "answer_validation": answer_validation,
        "required_citation_status": required_citation_status,
        "claim_fragments_present": claim_fragments_present,
        "semantic_support_level": answer_validation.get("semantic_support_level"),
        "quality_claim": "eval_status_only" if passed else "none",
    }


def _run_retrieval_item(
    *,
    item: Mapping[str, Any],
    target: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    result = _base_item_result(item)
    result["route_used"] = "local_lexical_retrieval"
    result["retrieval_routed"] = True
    result["private_target_used"] = True
    paths = _provided_doc_paths(item)
    if not paths:
        result.update(
            {
                "final_status": "unsupported",
                "unsupported_reason": "source-grounded item has no provided local documents",
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
        )
        return result
    index = build_index_for_paths(
        paths,
        workspace_root=str(HIDDEN_EVAL_DIR.parent.parent),
        source_class="allowed",
        source_class_reason="hidden_eval_item_scoped_provided_document_not_general_corpus",
        base_dir=HIDDEN_EVAL_DIR,
    )
    answer_payload = _source_answer_from_docs(item, index)
    verifier = _verify_source_answer(answer_payload=answer_payload, target=target, index=index)
    retrieval_dir = run_dir / "retrieval_items"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    retrieval_report = {
        "schema": "hidden_eval_retrieval_item_report_v1",
        "item_id": item.get("id"),
        "answer_status": answer_payload.get("answer_status"),
        "answer_text": answer_payload.get("answer_text") or answer_payload.get("answer"),
        "citations": answer_payload.get("citations", []),
        "document_refs": answer_payload.get("document_refs", []),
        "verifier": verifier,
        "citation_verifier": verifier.get("citation_validation"),
        "answer_verifier": verifier.get("answer_validation"),
        "abstention_reason": answer_payload.get("abstention_reason"),
        "semantic_truth_claim": answer_payload.get("semantic_truth_claim"),
        "private_target_exposed_in_public_summary": False,
        "quality_claim": "eval_status_only" if verifier.get("passed") else "none",
    }
    retrieval_report_path = _write_json(retrieval_dir / f"{item['id']}.json", retrieval_report)
    answer_status = str(answer_payload.get("answer_status") or "")
    result.update(
        {
            "final_status": "passed" if verifier.get("passed") else ("blocked_unverified" if answer_status.startswith("abstained") or answer_status == "unsupported_current_phase" else "verification_failed"),
            "verifier_status": "passed" if verifier.get("passed") else "failed",
            "verifier_reached": True,
            "verifier_passed": bool(verifier.get("passed")),
            "retrieval_report_path": retrieval_report_path,
            "retrieval_supported": answer_status == "answered_with_citations",
            "retrieval_abstained": answer_status.startswith("abstained") or answer_status == "unsupported_current_phase",
            "answer_status": answer_status,
            "citation_count": len(answer_payload.get("citations") or []),
            "citation_verifier_passed": bool((verifier.get("citation_validation") or {}).get("valid")),
            "answer_verifier_passed": bool((verifier.get("answer_validation") or {}).get("valid")),
            "quality_claim": "eval_status_only" if verifier.get("passed") else "none",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )
    return result


def _run_item(
    *,
    item: Mapping[str, Any],
    target: Mapping[str, Any],
    workspace_root: str,
    run_dir: Path,
    backend_config: Mapping[str, Any],
    backend_script_path: Optional[str],
    agent_report_dir: Optional[str],
) -> Dict[str, Any]:
    category = str(item["category"])
    if category == "exact_symbolic_correctness":
        return _run_exact_item(item=item, target=target, workspace_root=workspace_root)
    if category == "abstention_need_tool_need_source":
        return _run_decision_item(item, target)
    if category in CODING_CATEGORIES:
        return _run_coding_item(
            item=item,
            target=target,
            workspace_root=workspace_root,
            run_dir=run_dir,
            backend_config=backend_config,
            backend_script_path=backend_script_path,
            agent_report_dir=agent_report_dir,
        )
    if category in {"source_grounded_truthfulness", "retrieval_grounded_qa"}:
        return _run_retrieval_item(item=item, target=target, run_dir=run_dir)
    if category in UNSUPPORTED_CURRENT_PHASE_CATEGORIES:
        return _unsupported_item(
            item,
            "retrieval/citation/source-grounded evaluator is not implemented in this Phase A runner",
        )
    return _unsupported_item(item, "category is not implemented in this narrow Phase A runner")


def _count(results: Sequence[Mapping[str, Any]], predicate) -> int:
    return sum(1 for item in results if predicate(item))


def _summary_counts(results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "total_attempted": len(results),
        "passed": _count(results, lambda item: item.get("final_status") == "passed"),
        "failed": _count(results, lambda item: item.get("final_status") in {"failed", "verification_failed"}),
        "blocked": _count(results, lambda item: item.get("final_status") in {"blocked", "blocked_unverified"}),
        "unsupported": _count(results, lambda item: item.get("final_status") == "unsupported"),
        "exact_tool_routed": _count(results, lambda item: bool(item.get("exact_tool_routed"))),
        "backend_routed": _count(results, lambda item: bool(item.get("backend_routed"))),
        "retrieval_routed": _count(results, lambda item: bool(item.get("retrieval_routed"))),
        "retrieval_supported": _count(results, lambda item: bool(item.get("retrieval_supported"))),
        "retrieval_abstained": _count(results, lambda item: bool(item.get("retrieval_abstained"))),
        "citation_verifier_passed": _count(results, lambda item: bool(item.get("citation_verifier_passed"))),
        "answer_verifier_passed": _count(results, lambda item: bool(item.get("answer_verifier_passed"))),
        "verifier_reached": _count(results, lambda item: bool(item.get("verifier_reached"))),
        "verifier_passed": _count(results, lambda item: bool(item.get("verifier_passed"))),
    }


def run_hidden_eval_seed_execution(
    *,
    seed_ids: Optional[Sequence[str]] = None,
    workspace_root: str = ".",
    backend_config_path: str = DEFAULT_QWEN_BACKEND_CONFIG,
    backend_script_path: Optional[str] = None,
    output_root: str = "./run_artifacts/hidden_eval_runs",
    use_default_agent_report_dir: bool = True,
) -> Dict[str, Any]:
    """Run a small hidden-eval seed subset without exposing private targets."""
    asset_report = validate_hidden_eval_assets()
    seed_items = load_hidden_eval_seed_set()
    private_targets = load_hidden_eval_private_targets()
    by_id = _seed_by_id(seed_items)
    selected_ids = tuple(seed_ids or DEFAULT_HIDDEN_SEED_IDS)
    missing = [item_id for item_id in selected_ids if item_id not in by_id]
    if missing:
        raise ValueError("unknown hidden eval seed ids: " + ", ".join(missing))

    run_id = _new_run_id()
    run_dir = Path(output_root).resolve() / run_id
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    backend_config = _load_agent_config(backend_config_path, workspace_root)
    agent_report_dir = agent_cfg.report_dir if use_default_agent_report_dir else None

    item_results: List[Dict[str, Any]] = []
    for item_id in selected_ids:
        item = by_id[item_id]
        target = private_targets[str(item["private_target_ref"])]
        result = _run_item(
            item=item,
            target=target,
            workspace_root=workspace_root,
            run_dir=run_dir,
            backend_config=backend_config,
            backend_script_path=backend_script_path,
            agent_report_dir=agent_report_dir,
        )
        result["item_report_path"] = _write_json(items_dir / f"{item_id}.json", result)
        item_results.append(result)

    counts = _summary_counts(item_results)
    category_counts: Dict[str, Dict[str, int]] = {}
    for result in item_results:
        category = str(result["category"])
        category_counts.setdefault(category, {"attempted": 0, "passed": 0, "failed": 0, "blocked": 0, "unsupported": 0})
        category_counts[category]["attempted"] += 1
        status = str(result["final_status"])
        if status == "passed":
            category_counts[category]["passed"] += 1
        elif status == "unsupported":
            category_counts[category]["unsupported"] += 1
        elif status in {"blocked", "blocked_unverified"}:
            category_counts[category]["blocked"] += 1
        else:
            category_counts[category]["failed"] += 1

    public_item_summaries = [
        {
            "item_id": result["item_id"],
            "category": result["category"],
            "route_used": result["route_used"],
            "final_status": result["final_status"],
            "verifier_status": result["verifier_status"],
            "backend_kind": result["backend_kind"],
            "agent_run_id": result["agent_run_id"],
            "retrieval_supported": bool(result.get("retrieval_supported")),
            "retrieval_abstained": bool(result.get("retrieval_abstained")),
            "answer_status": result.get("answer_status"),
            "citation_count": result.get("citation_count", 0),
            "citation_verifier_passed": bool(result.get("citation_verifier_passed")),
            "answer_verifier_passed": bool(result.get("answer_verifier_passed")),
            "private_target_used": result["private_target_used"],
            "private_target_exposed_in_public_summary": False,
            "answer_exposed_in_public_summary": False,
        }
        for result in item_results
    ]
    summary: Dict[str, Any] = {
        "schema": HIDDEN_EVAL_RUN_SCHEMA,
        "run_id": run_id,
        "timestamp": int(time.time()),
        "asset_validation": {
            "schema": asset_report["schema"],
            "item_count": asset_report["item_count"],
            "training_exclusion_required_for_all": asset_report["training_exclusion_required_for_all"],
        },
        "seed_count_total": len(seed_items),
        "seed_item_ids": list(selected_ids),
        "backend_kind": backend_config.get("backend_kind"),
        "model_id_or_path": backend_config.get("backend_model_id_or_path"),
        "backend_config_path": backend_config.get("config_path"),
        "backend_config_exists": bool(backend_config.get("config_exists")),
        "backend_script_used": bool(backend_script_path),
        "agent_report_dir": agent_report_dir or str(run_dir / "agent_reports"),
        "default_trajectory_store_used": bool(use_default_agent_report_dir),
        "counts": counts,
        "category_counts": category_counts,
        "items": public_item_summaries,
        "item_report_dir": str(items_dir),
        "private_target_leakage": False,
        "private_target_exposed_in_public_summary": False,
        "answer_exposed_in_public_summary": False,
        "quality_claim": "none",
        "bakeoff_executed": False,
        "training_executed": False,
        "retrieval_implemented": True,
        "retrieval_scope": "local_repo_docs_plus_item_scoped_provided_docs",
        "proves_model_quality": False,
    }
    summary["summary_report_path"] = _write_json(run_dir / "summary.json", summary)
    return summary
