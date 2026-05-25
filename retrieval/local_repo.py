"""Deterministic local repo-document retrieval and citation checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


INDEX_SCHEMA = "local_repo_docs_retrieval_index_v1"
CITATION_SCHEMA = "retrieval_citation_v1"
DEFAULT_INDEX_PATH = "./run_artifacts/retrieval_index/local_repo_docs_index_v1.json"
ALLOWED_SOURCE_CLASS = "allowed"
ALLOWED_EXTENSIONS = {".md", ".txt", ".json", ".jsonl"}
ALLOWED_DOC_ROOTS = (
    "SYSTEM_MAP.md",
    "backend_integration",
    "retrieval_design",
    "model_selection",
    "data_governance",
    "trajectory_quality",
)
ALLOWED_HIDDEN_PUBLIC_SPECS = {
    "evals/hidden/hidden_eval_spec_v1.md",
    "evals/hidden/hidden_eval_spec_v1.json",
}
EXCLUDED_PREFIXES = (
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "data_cache",
    "tokenizer_data",
    "run_artifacts",
    "checkpoints",
    "sft_checkpoints",
    "dpo_checkpoints",
    "distill_checkpoints",
    "improved_checkpoints",
    "quantized",
    "eval_results",
    "evals/hidden/private",
    "evals/hidden/fixtures",
)
EXCLUDED_NAME_PARTS = (
    "__pycache__",
    ".pyc",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".onnx",
)
REQUIRED_CITATION_FIELDS = (
    "citation_id",
    "source_class",
    "source_id",
    "document_ref",
    "locator_type",
    "locator",
    "support_kind",
    "support_snippet_hash",
)
ALLOWED_LOCATOR_TYPES = {"line_range", "json_pointer", "section_heading"}


def _repo_rel(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _is_prefix(path_ref: str, prefix: str) -> bool:
    return path_ref == prefix or path_ref.startswith(prefix.rstrip("/") + "/")


def classify_repo_path(path: Path, workspace_root: str = ".") -> Tuple[str, str]:
    """Return source class and reason for a repo-relative path."""
    root = Path(workspace_root).resolve()
    try:
        rel = _repo_rel(path, root)
    except ValueError:
        return "blocked", "path_outside_workspace"
    lowered = rel.lower()
    for marker in EXCLUDED_NAME_PARTS:
        if marker in lowered:
            return "blocked", f"excluded_name_marker:{marker}"
    for prefix in EXCLUDED_PREFIXES:
        if _is_prefix(rel, prefix):
            return "benchmark_excluded" if prefix.startswith("evals/hidden") else "blocked", f"excluded_prefix:{prefix}"
    if rel in ALLOWED_HIDDEN_PUBLIC_SPECS:
        return ALLOWED_SOURCE_CLASS, "allowed_hidden_public_spec"
    for root_ref in ALLOWED_DOC_ROOTS:
        if rel == root_ref or rel.startswith(root_ref.rstrip("/") + "/"):
            return ALLOWED_SOURCE_CLASS, f"allowed_root:{root_ref}"
    return "unknown", "not_in_phase_a_retrieval_allowlist"


def discover_index_candidates(workspace_root: str = ".") -> Tuple[List[Path], List[Dict[str, Any]]]:
    root = Path(workspace_root).resolve()
    candidates: List[Path] = []
    skips: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        try:
            current_rel = _repo_rel(current, root)
        except ValueError:
            continue
        if current_rel == ".":
            current_rel = ""
        kept_dirs = []
        for dirname in dirnames:
            rel_dir = f"{current_rel}/{dirname}".strip("/")
            if any(_is_prefix(rel_dir, prefix) for prefix in EXCLUDED_PREFIXES):
                skips.append(
                    {
                        "document_ref": rel_dir,
                        "source_class": "blocked",
                        "skip_reason": "excluded_directory_pruned",
                    }
                )
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in filenames:
            path = current / filename
            try:
                rel = _repo_rel(path, root)
            except ValueError:
                continue
            source_class, reason = classify_repo_path(path, workspace_root)
            if source_class == ALLOWED_SOURCE_CLASS and path.suffix.lower() in ALLOWED_EXTENSIONS:
                candidates.append(path)
            else:
                if (
                    path.suffix.lower() in ALLOWED_EXTENSIONS
                    or "evals/hidden" in rel
                    or "run_artifacts" in rel
                    or "data_cache" in rel
                ):
                    skips.append(
                        {
                            "document_ref": rel,
                            "source_class": source_class,
                            "skip_reason": reason if path.suffix.lower() in ALLOWED_EXTENSIONS else "extension_not_indexed",
                        }
                    )
    return sorted(candidates), skips


def _line_chunks(text: str, *, max_lines: int = 8) -> List[Tuple[str, str, str]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, str, str]] = []
    start = 0
    while start < len(lines):
        while start < len(lines) and not lines[start].strip():
            start += 1
        if start >= len(lines):
            break
        end = min(len(lines), start + max_lines)
        while end < len(lines) and lines[end].strip() and end - start < max_lines:
            end += 1
        chunk_text = "\n".join(lines[start:end]).strip()
        if chunk_text:
            chunks.append(("line_range", f"{start + 1}-{end}", chunk_text))
        start = end + 1
    return chunks


def _json_chunks(text: str) -> List[Tuple[str, str, str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _line_chunks(text)
    chunks: List[Tuple[str, str, str]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            chunks.append(("json_pointer", "/" + str(key).replace("~", "~0").replace("/", "~1"), json.dumps(value, sort_keys=True)))
    elif isinstance(payload, list):
        for index, value in enumerate(payload[:100]):
            chunks.append(("json_pointer", f"/{index}", json.dumps(value, sort_keys=True)))
    else:
        chunks.append(("json_pointer", "", json.dumps(payload, sort_keys=True)))
    return chunks


def _source_id(document_ref: str) -> str:
    return "src_" + _sha256_text(document_ref)[:16]


def build_index_for_paths(
    paths: Sequence[Path],
    *,
    workspace_root: str = ".",
    source_class: str = ALLOWED_SOURCE_CLASS,
    source_class_reason: str = "explicit_allowed_paths",
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(workspace_root).resolve()
    chunks: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []
    for path in paths:
        resolved = Path(path).resolve()
        if base_dir is not None:
            document_ref = resolved.relative_to(base_dir.resolve()).as_posix()
        else:
            document_ref = _repo_rel(resolved, root)
        text = resolved.read_text(encoding="utf-8", errors="replace")
        source_id = _source_id(document_ref)
        documents.append(
            {
                "source_id": source_id,
                "document_ref": document_ref,
                "source_class": source_class,
                "source_class_reason": source_class_reason,
                "content_hash": _sha256_text(text),
                "size_bytes": resolved.stat().st_size,
            }
        )
        raw_chunks = _json_chunks(text) if resolved.suffix.lower() in {".json", ".jsonl"} else _line_chunks(text)
        for index, (locator_type, locator, chunk_text) in enumerate(raw_chunks):
            snippet_hash = _sha256_text(chunk_text)
            chunks.append(
                {
                    "chunk_id": f"{source_id}#chunk-{index}",
                    "source_id": source_id,
                    "document_ref": document_ref,
                    "source_class": source_class,
                    "locator_type": locator_type,
                    "locator": locator,
                    "content": chunk_text,
                    "support_snippet_hash": snippet_hash,
                    "title": document_ref,
                }
            )
    return {
        "schema": INDEX_SCHEMA,
        "built_at": int(time.time()),
        "retrieval_mode": "deterministic_lexical",
        "source_scope": "explicit_paths" if base_dir is not None else "repo_authored_docs_and_manifests_only",
        "documents": documents,
        "chunks": chunks,
        "skip_report": [],
        "quality_claim": "none",
        "proves_truth": False,
    }


def build_local_repo_docs_index(
    *,
    workspace_root: str = ".",
    output_path: str = DEFAULT_INDEX_PATH,
) -> Dict[str, Any]:
    candidates, skips = discover_index_candidates(workspace_root)
    index = build_index_for_paths(candidates, workspace_root=workspace_root)
    index["skip_report"] = skips
    index["excluded_prefixes"] = list(EXCLUDED_PREFIXES)
    index["normal_index_excludes_hidden_eval_fixtures"] = True
    out = Path(output_path)
    if not out.is_absolute():
        out = Path(workspace_root).resolve() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    index["index_path"] = str(out.resolve())
    return index


def load_index(index_path: str = DEFAULT_INDEX_PATH, *, workspace_root: str = ".") -> Dict[str, Any]:
    path = Path(index_path)
    if not path.is_absolute():
        path = Path(workspace_root).resolve() / path
    return json.loads(path.read_text(encoding="utf-8"))


def make_citation(chunk: Mapping[str, Any], citation_index: int = 1) -> Dict[str, Any]:
    return {
        "schema": CITATION_SCHEMA,
        "citation_id": f"c{citation_index}",
        "source_class": chunk["source_class"],
        "source_id": chunk["source_id"],
        "document_ref": chunk["document_ref"],
        "locator_type": chunk["locator_type"],
        "locator": chunk["locator"],
        "support_kind": "lexical_evidence",
        "support_snippet_hash": chunk["support_snippet_hash"],
    }


def search_index(
    query: str,
    *,
    index: Optional[Mapping[str, Any]] = None,
    index_path: str = DEFAULT_INDEX_PATH,
    workspace_root: str = ".",
    top_k: int = 5,
) -> Dict[str, Any]:
    if index is None:
        index = load_index(index_path, workspace_root=workspace_root)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {
            "schema": "retrieval_search_result_v1",
            "query": query,
            "status": "abstain_no_query_terms",
            "matches": [],
            "citations": [],
            "abstention": abstain_for_missing_evidence(query, reason="empty_query"),
        }
    scored = []
    for chunk in index.get("chunks", []):
        if chunk.get("source_class") != ALLOWED_SOURCE_CLASS:
            continue
        text_tokens = _tokenize(str(chunk.get("content", "")) + " " + str(chunk.get("document_ref", "")))
        token_set = set(text_tokens)
        score = sum(2 if token in token_set else 0 for token in query_tokens)
        score += sum(str(chunk.get("content", "")).lower().count(token) for token in query_tokens)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("document_ref")), str(item[1].get("locator"))))
    matches = [
        {
            "score": score,
            "chunk_id": chunk["chunk_id"],
            "document_ref": chunk["document_ref"],
            "locator_type": chunk["locator_type"],
            "locator": chunk["locator"],
            "content_preview": str(chunk["content"])[:320],
            "citation": make_citation(chunk, index + 1),
        }
        for index, (score, chunk) in enumerate(scored[:top_k])
    ]
    return {
        "schema": "retrieval_search_result_v1",
        "query": query,
        "status": "evidence_found" if matches else "abstain_no_evidence",
        "matches": matches,
        "citations": [match["citation"] for match in matches],
        "abstention": None if matches else abstain_for_missing_evidence(query, reason="no_allowed_evidence_found"),
        "quality_claim": "none",
        "proves_truth": False,
    }


def _find_chunk(index: Mapping[str, Any], citation: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for chunk in index.get("chunks", []):
        if (
            chunk.get("source_id") == citation.get("source_id")
            and chunk.get("document_ref") == citation.get("document_ref")
            and chunk.get("locator_type") == citation.get("locator_type")
            and chunk.get("locator") == citation.get("locator")
        ):
            return chunk
    return None


def validate_citation(citation: Mapping[str, Any], index: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [field for field in REQUIRED_CITATION_FIELDS if field not in citation]
    if missing:
        return {
            "valid": False,
            "failure_reason": "missing_required_field:" + ",".join(missing),
            "source_policy_status": "not_checked",
            "locator_resolution_status": "not_checked",
        }
    if citation.get("source_class") != ALLOWED_SOURCE_CLASS:
        return {
            "valid": False,
            "failure_reason": "source_class_not_allowed",
            "source_policy_status": "blocked",
            "locator_resolution_status": "not_checked",
        }
    if citation.get("locator_type") not in ALLOWED_LOCATOR_TYPES:
        return {
            "valid": False,
            "failure_reason": "locator_type_not_allowed",
            "source_policy_status": "allowed",
            "locator_resolution_status": "failed",
        }
    chunk = _find_chunk(index, citation)
    if not chunk:
        return {
            "valid": False,
            "failure_reason": "unresolvable_locator",
            "source_policy_status": "allowed",
            "locator_resolution_status": "failed",
        }
    if chunk.get("support_snippet_hash") != citation.get("support_snippet_hash"):
        return {
            "valid": False,
            "failure_reason": "support_snippet_hash_mismatch",
            "source_policy_status": "allowed",
            "locator_resolution_status": "resolved",
        }
    document_ref = str(citation.get("document_ref", ""))
    if document_ref.startswith("evals/hidden/private") or "/private/" in document_ref:
        return {
            "valid": False,
            "failure_reason": "hidden_private_source_not_citable",
            "source_policy_status": "blocked",
            "locator_resolution_status": "resolved",
        }
    return {
        "valid": True,
        "failure_reason": None,
        "source_policy_status": "allowed",
        "locator_resolution_status": "resolved",
        "resolved_chunk_id": chunk.get("chunk_id"),
    }


def validate_citation_bundle(citations: Sequence[Mapping[str, Any]], index: Mapping[str, Any]) -> Dict[str, Any]:
    results = [validate_citation(citation, index) for citation in citations]
    valid = bool(citations) and all(item["valid"] for item in results)
    return {
        "schema": "retrieval_citation_validation_v1",
        "valid": valid,
        "citation_count": len(citations),
        "results": results,
        "failure_reason": None if valid else "missing_or_invalid_citations",
        "quality_claim": "none",
        "proves_truth": False,
    }


def abstain_for_missing_evidence(query: str, *, reason: str) -> Dict[str, Any]:
    return {
        "schema": "retrieval_abstention_v1",
        "status": "abstain",
        "reason": reason,
        "answer": "I do not have allowed local evidence for that claim in the current retrieval scope.",
        "query": query,
        "quality_claim": "none",
    }


def citation_check_for_query(
    query: str,
    *,
    index_path: str = DEFAULT_INDEX_PATH,
    workspace_root: str = ".",
) -> Dict[str, Any]:
    index = load_index(index_path, workspace_root=workspace_root)
    search = search_index(query, index=index)
    validation = validate_citation_bundle(search.get("citations", []), index)
    return {
        "schema": "retrieval_citation_check_report_v1",
        "query": query,
        "search_status": search["status"],
        "citation_validation": validation,
        "citations": search.get("citations", []),
        "quality_claim": "none",
        "proves_truth": False,
    }
