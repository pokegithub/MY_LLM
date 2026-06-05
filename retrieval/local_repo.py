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
ANSWER_SCHEMA = "retrieval_cited_answer_v1"
ANSWER_VALIDATION_SCHEMA = "retrieval_cited_answer_validation_v1"
CLAIM_SUPPORT_SCHEMA = "retrieval_claim_support_report_v1"
DEFAULT_INDEX_PATH = "./run_artifacts/retrieval_index/local_repo_docs_index_v1.json"
ALLOWED_SOURCE_CLASS = "allowed"
ALLOWED_EXTENSIONS = {".md", ".txt", ".json", ".jsonl"}
ANSWERED_STATUSES = {"answered_with_citations", "partial_answer_with_caveats"}
ABSTENTION_STATUSES = {
    "abstained_no_evidence",
    "abstained_out_of_scope",
    "blocked_source_policy",
    "unsupported_current_phase",
}
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
OUT_OF_SCOPE_QUERY_MARKERS = (
    "today",
    "latest",
    "current news",
    "world news",
    "breaking news",
    "weather",
    "stock price",
    "exchange rate",
    "search the web",
    "internet",
)
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
PASSING_CLAIM_SUPPORT_LEVELS = {"direct_quote_or_near_quote", "lexical_overlap"}
PARTIAL_CLAIM_SUPPORT_LEVELS = {"locator_only_weak"}


def _repo_rel(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _query_terms(text: str) -> List[str]:
    return [token for token in _tokenize(text) if len(token) > 2 and token not in QUERY_STOPWORDS]


def _claim_terms(text: str) -> List[str]:
    text = re.sub(r"\[c\d+\]", " ", text, flags=re.IGNORECASE)
    return [token for token in _query_terms(text) if not token.startswith("c")]


def _command_phrases(text: str) -> List[str]:
    return re.findall(r"\b[a-z][a-z0-9_]*-[a-z0-9_-]+\b", text.lower())


def _is_prefix(path_ref: str, prefix: str) -> bool:
    return path_ref == prefix or path_ref.startswith(prefix.rstrip("/") + "/")


def _is_document_ref_blocked_for_citation(document_ref: str) -> Tuple[bool, Optional[str]]:
    normalized = document_ref.replace("\\", "/").strip().lower()
    if not normalized:
        return True, "empty_document_ref"
    if normalized.startswith("evals/hidden/private") or normalized.startswith("private/") or "/private/" in normalized:
        return True, "hidden_private_source_not_citable"
    for prefix in EXCLUDED_PREFIXES:
        if _is_prefix(normalized, prefix.lower()):
            return True, f"excluded_prefix_not_citable:{prefix}"
    for marker in EXCLUDED_NAME_PARTS:
        if marker in normalized:
            return True, f"excluded_name_marker_not_citable:{marker}"
    return False, None


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
    useful_query_tokens = _query_terms(query)
    command_phrases = _command_phrases(query)
    for chunk in index.get("chunks", []):
        if chunk.get("source_class") != ALLOWED_SOURCE_CLASS:
            continue
        content = str(chunk.get("content", ""))
        document_ref = str(chunk.get("document_ref", ""))
        content_lower = content.lower()
        document_lower = document_ref.lower().replace("/", " ").replace("_", " ")
        if command_phrases and not any(
            phrase in content_lower or phrase in document_ref.lower()
            for phrase in command_phrases
        ):
            continue
        content_tokens = _tokenize(content)
        document_tokens = _tokenize(document_lower)
        token_set = set(content_tokens + document_tokens)
        document_token_set = set(document_tokens)
        score = sum(2 if token in token_set else 0 for token in query_tokens)
        if command_phrases:
            score += 120
        score += sum(content_lower.count(token) for token in query_tokens)
        document_match_count = sum(1 for token in useful_query_tokens if token in document_token_set)
        score += document_match_count * 10
        if useful_query_tokens and document_match_count == len(useful_query_tokens):
            score += 75
        if document_ref.endswith(".md") and document_match_count:
            score += 10
        score += sum(document_lower.count(token) * 3 for token in useful_query_tokens)
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
    blocked, block_reason = _is_document_ref_blocked_for_citation(str(citation.get("document_ref", "")))
    if blocked:
        return {
            "valid": False,
            "failure_reason": block_reason,
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


def _query_is_out_of_scope(query: str) -> bool:
    lowered = query.lower()
    return any(marker in lowered for marker in OUT_OF_SCOPE_QUERY_MARKERS)


def _answer_abstention_payload(
    query: str,
    *,
    answer_status: str,
    reason: str,
    source_policy_status: str = "not_checked",
) -> Dict[str, Any]:
    return {
        "schema": ANSWER_SCHEMA,
        "query": query,
        "answer_status": answer_status,
        "answer_text": "",
        "citations": [],
        "unsupported_claims": [query] if query.strip() else [],
        "abstention_reason": reason,
        "source_policy_status": source_policy_status,
        "quality_claim": "none",
        "semantic_truth_claim": "limited_or_none",
        "semantic_support_level": "none",
        "search_status": "not_attempted" if answer_status == "unsupported_current_phase" else "abstained",
        "answer_assembly": "abstention_policy_v1",
        "proves_truth": False,
    }


def _clean_snippet_for_answer(
    content: str,
    *,
    max_chars: int = 260,
    focus_phrases: Optional[Sequence[str]] = None,
) -> str:
    parts: List[str] = []
    focused_parts: List[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if line:
            if focus_phrases and any(phrase in line.lower() for phrase in focus_phrases):
                focused_parts.append(line)
            parts.append(line)
        if not focus_phrases and len(" ".join(parts)) >= max_chars:
            break
    if focused_parts:
        parts = focused_parts
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _chunk_by_citation(index: Mapping[str, Any], citation: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    return _find_chunk(index, citation)


def _citation_answer_markers_present(answer_text: str, citations: Sequence[Mapping[str, Any]]) -> bool:
    if not answer_text.strip() or not citations:
        return False
    return any(f"[{citation.get('citation_id')}]" in answer_text for citation in citations)


def _strip_citation_markers(text: str) -> str:
    return re.sub(r"\[c\d+\]", "", text, flags=re.IGNORECASE).strip()


def _normalize_claim_text(text: str) -> str:
    cleaned = _strip_citation_markers(text)
    cleaned = re.sub(r"[`*_#>\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().lower()


def _split_uncited_text_into_claims(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+|(?:^|\n)\s*[-*]\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_answer_claims(answer_text: str) -> List[Dict[str, Any]]:
    """Split a cited answer into deterministic, checkable claim records.

    Citation-marked spans are kept intact so extractive answers such as
    "sentence one. sentence two [c1]" are treated as one cited evidence-bound
    claim instead of a free-floating uncited first sentence.
    """
    text = str(answer_text or "").strip()
    if not text:
        return []
    claims: List[Dict[str, Any]] = []
    cursor = 0
    marker_span = re.compile(r".*?\[(?:c\d+)(?:\s*,\s*c\d+)*\]", re.IGNORECASE | re.DOTALL)
    for match in marker_span.finditer(text):
        before = text[cursor:match.start()].strip()
        for piece in _split_uncited_text_into_claims(before):
            claims.append(_claim_record(len(claims) + 1, piece))
        claims.append(_claim_record(len(claims) + 1, match.group(0).strip()))
        cursor = match.end()
    for piece in _split_uncited_text_into_claims(text[cursor:].strip()):
        claims.append(_claim_record(len(claims) + 1, piece))
    return claims


def _claim_record(index: int, claim_text: str) -> Dict[str, Any]:
    citation_ids = re.findall(r"\[(c\d+)\]", claim_text, flags=re.IGNORECASE)
    normalized_text = _normalize_claim_text(claim_text)
    requires_citation = bool(_claim_terms(normalized_text))
    return {
        "claim_id": f"claim_{index}",
        "claim_text": claim_text.strip(),
        "requires_citation": requires_citation,
        "support_status": "not_checked",
        "supporting_citation_ids": [citation_id.lower() for citation_id in citation_ids],
        "unsupported_reason": None,
    }


def _query_coverage_from_citations(
    query: str,
    citations: Sequence[Mapping[str, Any]],
    index: Mapping[str, Any],
) -> Dict[str, Any]:
    query_terms = set(_query_terms(query))
    if not query_terms:
        return {"matched_terms": [], "query_terms": [], "coverage_ratio": 0.0, "direct_enough": False}
    evidence_terms = set()
    for citation in citations:
        chunk = _chunk_by_citation(index, citation)
        if not chunk:
            continue
        evidence_terms.update(_query_terms(str(chunk.get("content", ""))))
        evidence_terms.update(_query_terms(str(chunk.get("document_ref", ""))))
    matched = sorted(query_terms.intersection(evidence_terms))
    coverage_ratio = len(matched) / len(query_terms)
    return {
        "matched_terms": matched,
        "query_terms": sorted(query_terms),
        "coverage_ratio": coverage_ratio,
        "direct_enough": coverage_ratio >= 0.6,
    }


def assemble_cited_answer(
    query: str,
    *,
    index: Optional[Mapping[str, Any]] = None,
    index_path: str = DEFAULT_INDEX_PATH,
    workspace_root: str = ".",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Assemble a short evidence-bound answer or abstain.

    This is intentionally extractive/template-based. It does not ask a model to
    synthesize unsupported claims, and it reports only lexical/locator support.
    """
    if _query_is_out_of_scope(query):
        return _answer_abstention_payload(
            query,
            answer_status="unsupported_current_phase",
            reason="requires_external_or_current_source",
            source_policy_status="out_of_scope",
        )
    if index is None:
        index = load_index(index_path, workspace_root=workspace_root)
    command_phrases = _command_phrases(query)
    search = search_index(query, index=index, top_k=top_k)
    if search.get("status") != "evidence_found":
        return _answer_abstention_payload(
            query,
            answer_status="abstained_no_evidence",
            reason=(search.get("abstention") or {}).get("reason", "no_allowed_evidence_found"),
            source_policy_status="allowed",
        )
    citations = list(search.get("citations") or [])
    coverage = _query_coverage_from_citations(query, citations, index)
    if not coverage.get("direct_enough"):
        payload = _answer_abstention_payload(
            query,
            answer_status="abstained_no_evidence",
            reason="weak_lexical_evidence",
            source_policy_status="allowed",
        )
        payload["claim_support_check"] = coverage
        return payload
    validation = validate_citation_bundle(citations, index)
    if not validation.get("valid"):
        return _answer_abstention_payload(
            query,
            answer_status="blocked_source_policy",
            reason=str(validation.get("failure_reason") or "invalid_citations"),
            source_policy_status="blocked",
        )
    answer_parts: List[str] = []
    used_citations: List[Mapping[str, Any]] = []
    answer_ordered_citations = sorted(
        citations[:top_k],
        key=lambda citation: (0 if str(citation.get("document_ref", "")).endswith(".md") else 1, str(citation.get("document_ref", ""))),
    )
    for citation in answer_ordered_citations:
        chunk = _chunk_by_citation(index, citation)
        if not chunk:
            continue
        cleaned = _clean_snippet_for_answer(str(chunk.get("content", "")), focus_phrases=command_phrases)
        if cleaned:
            answer_parts.append(f"{cleaned} [{citation.get('citation_id')}]")
            used_citations.append(citation)
        if len(answer_parts) >= 2:
            break
    if not answer_parts:
        return _answer_abstention_payload(
            query,
            answer_status="abstained_no_evidence",
            reason="retrieved_chunks_had_no_reusable_text",
            source_policy_status="allowed",
        )
    payload = {
        "schema": ANSWER_SCHEMA,
        "query": query,
        "answer_status": "answered_with_citations",
        "answer_text": " ".join(answer_parts),
        "citations": used_citations,
        "unsupported_claims": [],
        "abstention_reason": None,
        "source_policy_status": "allowed",
        "quality_claim": "none",
        "semantic_truth_claim": "limited_or_none",
        "semantic_support_level": "lexical_or_locator_only",
        "search_status": search.get("status"),
        "answer_assembly": "extractive_template_v1",
        "proves_truth": False,
    }
    payload["answer_verification"] = validate_cited_answer(payload, index)
    payload["unsupported_claims"] = [
        claim.get("claim_text")
        for claim in payload["answer_verification"].get("unsupported_claims", [])
    ]
    return payload


def _citation_lookup(citations: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(citation.get("citation_id", "")).lower(): citation for citation in citations}


def _contains_unnegated_word(text: str, words: Sequence[str]) -> bool:
    for word in words:
        pattern = re.compile(rf"\b{re.escape(word)}\b")
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 24):match.start()]
            if not re.search(r"\b(no|not|never|without|missing|unsupported|unavailable|disabled|failed|blocked)\b", prefix):
                return True
    return False


def _contradiction_subject_terms(text: str) -> set:
    generic = {
        "answer",
        "answers",
        "available",
        "command",
        "docs",
        "document",
        "enabled",
        "exists",
        "implemented",
        "local",
        "phase",
        "question",
        "questions",
        "ready",
        "repo",
        "runtime",
        "source",
        "supported",
    }
    return set(_claim_terms(text)) - generic


def _conservative_contradiction_reason(claim_normalized: str, evidence_normalized: str) -> Optional[str]:
    """Return a narrow lexical contradiction reason, or None.

    This is intentionally conservative. It catches a few high-risk overclaims
    without pretending to perform semantic entailment.
    """
    absolute_markers = ("always", "guaranteed", "guarantees", "instantly", "immediately")
    conditional_or_delay_markers = (
        "may",
        "might",
        "optional",
        "manually",
        "not guaranteed",
        "not always",
        "delayed",
        "delay",
    )
    if (
        any(marker in claim_normalized for marker in absolute_markers)
        and any(marker in evidence_normalized for marker in conditional_or_delay_markers)
    ):
        return "absolute_claim_conflicts_with_conditional_or_delayed_evidence"

    positive_terms = (
        "available",
        "enabled",
        "exists",
        "implemented",
        "passed",
        "ready",
        "supported",
        "success",
        "succeeded",
    )
    negative_phrases = (
        "blocked",
        "disabled",
        "did not implement",
        "does not exist",
        "failed",
        "must not",
        "not available",
        "not enabled",
        "not implemented",
        "not ready",
        "not supported",
        "unsupported",
        "unavailable",
    )
    if _contains_unnegated_word(claim_normalized, positive_terms):
        claim_subject_terms = _contradiction_subject_terms(claim_normalized)
        for evidence_sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence_normalized):
            if not any(phrase in evidence_sentence for phrase in negative_phrases):
                continue
            sentence_subject_terms = _contradiction_subject_terms(evidence_sentence)
            if claim_subject_terms.intersection(sentence_subject_terms):
                return "positive_claim_conflicts_with_negative_evidence"
    return None


def _direct_or_lexical_claim_support(
    claim_text: str,
    citation_ids: Sequence[str],
    citations: Sequence[Mapping[str, Any]],
    index: Mapping[str, Any],
    citation_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    lookup = _citation_lookup(citations)
    if not citation_ids:
        return {
            "support_status": "citation_missing",
            "unsupported_reason": "missing_claim_citation",
            "supporting_citation_ids": [],
            "matched_terms": [],
            "coverage_ratio": 0.0,
        }

    evidence_texts: List[str] = []
    valid_citation_ids: List[str] = []
    invalid_citation_ids: List[str] = []
    for citation_id in citation_ids:
        citation = lookup.get(citation_id.lower())
        validation = citation_results.get(citation_id.lower())
        if not citation or not validation or not validation.get("valid"):
            invalid_citation_ids.append(citation_id)
            continue
        chunk = _chunk_by_citation(index, citation)
        if not chunk:
            invalid_citation_ids.append(citation_id)
            continue
        valid_citation_ids.append(citation_id.lower())
        evidence_texts.append(str(chunk.get("content", "")))
        evidence_texts.append(str(chunk.get("document_ref", "")))

    if invalid_citation_ids:
        return {
            "support_status": "unsupported",
            "unsupported_reason": "invalid_claim_citation:" + ",".join(sorted(invalid_citation_ids)),
            "supporting_citation_ids": valid_citation_ids,
            "matched_terms": [],
            "coverage_ratio": 0.0,
        }
    if not evidence_texts:
        return {
            "support_status": "unsupported",
            "unsupported_reason": "unresolved_claim_evidence",
            "supporting_citation_ids": valid_citation_ids,
            "matched_terms": [],
            "coverage_ratio": 0.0,
        }

    claim_normalized = _normalize_claim_text(claim_text)
    evidence_normalized = _normalize_claim_text(" ".join(evidence_texts))
    claim_terms = set(_claim_terms(claim_normalized))
    evidence_terms = set(_claim_terms(evidence_normalized))
    matched = sorted(claim_terms.intersection(evidence_terms))
    coverage_ratio = len(matched) / len(claim_terms) if claim_terms else 1.0
    contradiction_reason = _conservative_contradiction_reason(claim_normalized, evidence_normalized)

    if contradiction_reason:
        status = "contradicted"
        reason = contradiction_reason
    elif claim_normalized and claim_normalized in evidence_normalized:
        status = "direct_quote_or_near_quote"
        reason = None
    elif claim_terms and coverage_ratio >= 0.6 and len(matched) >= min(2, len(claim_terms)):
        status = "lexical_overlap"
        reason = None
    elif valid_citation_ids:
        status = "locator_only_weak"
        reason = "weak_lexical_support"
    else:
        status = "unsupported"
        reason = "no_valid_claim_support"

    return {
        "support_status": status,
        "unsupported_reason": reason,
        "supporting_citation_ids": valid_citation_ids,
        "matched_terms": matched[:20],
        "coverage_ratio": coverage_ratio,
    }


def claim_support_report(
    *,
    answer_text: str,
    citations: Sequence[Mapping[str, Any]],
    index: Mapping[str, Any],
    citation_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    claims = extract_answer_claims(answer_text)
    citation_results: Dict[str, Mapping[str, Any]] = {}
    for citation, validation in zip(citations, citation_validation.get("results", [])):
        citation_results[str(citation.get("citation_id", "")).lower()] = validation

    checked_claims: List[Dict[str, Any]] = []
    for claim in claims:
        updated = dict(claim)
        if not updated.get("requires_citation"):
            updated["support_status"] = "not_checked"
            checked_claims.append(updated)
            continue
        support = _direct_or_lexical_claim_support(
            str(updated.get("claim_text", "")),
            list(updated.get("supporting_citation_ids") or []),
            citations,
            index,
            citation_results,
        )
        updated.update(support)
        checked_claims.append(updated)

    required_claims = [claim for claim in checked_claims if claim.get("requires_citation")]
    supported_claims = [
        claim for claim in required_claims
        if claim.get("support_status") in PASSING_CLAIM_SUPPORT_LEVELS
    ]
    partially_supported_claims = [
        claim for claim in required_claims
        if claim.get("support_status") in PARTIAL_CLAIM_SUPPORT_LEVELS
    ]
    contradicted_claims = [
        claim for claim in required_claims
        if claim.get("support_status") == "contradicted"
    ]
    missing_citation_claims = [
        claim for claim in required_claims
        if claim.get("support_status") == "citation_missing"
    ]
    strictly_unsupported_claims = [
        claim for claim in required_claims
        if claim.get("support_status") == "unsupported"
    ]
    unsupported_claims = [
        claim for claim in required_claims
        if claim.get("support_status") not in PASSING_CLAIM_SUPPORT_LEVELS
    ]
    support_levels = {str(claim.get("support_status")) for claim in required_claims}
    if not required_claims:
        aggregate = "not_checked"
        answer_support_quality = "not_checked"
    elif contradicted_claims:
        aggregate = "contradicted"
        answer_support_quality = "contradicted"
    elif strictly_unsupported_claims or missing_citation_claims:
        aggregate = "unsupported"
        answer_support_quality = "unsupported"
    elif partially_supported_claims:
        aggregate = "partial"
        answer_support_quality = "partial"
    elif "direct_quote_or_near_quote" in support_levels and len(support_levels) == 1:
        aggregate = "direct_quote_or_near_quote"
        answer_support_quality = "fully_supported"
    else:
        aggregate = "lexical_overlap"
        answer_support_quality = "fully_supported"
    return {
        "schema": CLAIM_SUPPORT_SCHEMA,
        "claim_count": len(claims),
        "claims_checked": len(required_claims),
        "claims_supported": len(supported_claims),
        "claims_partially_supported": len(partially_supported_claims),
        "claims_unsupported": len(strictly_unsupported_claims),
        "claims_contradicted": len(contradicted_claims),
        "claims_missing_citations": len(missing_citation_claims),
        "contradiction_detected": bool(contradicted_claims),
        "answer_support_quality": answer_support_quality,
        "claim_support_level": aggregate,
        "unsupported_claims": unsupported_claims,
        "partially_supported_claims": partially_supported_claims,
        "contradicted_claims": contradicted_claims,
        "missing_citation_claims": missing_citation_claims,
        "claims": checked_claims,
        "semantic_support_level": "lexical_or_locator_only",
        "semantic_truth_claim": "limited_or_none",
        "quality_claim": "none",
    }


def validate_cited_answer(answer_payload: Mapping[str, Any], index: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(answer_payload.get("answer_status") or "")
    answer_text = str(answer_payload.get("answer_text") or answer_payload.get("answer") or "")
    citations = list(answer_payload.get("citations") or [])
    failure_reasons: List[str] = []

    if answer_payload.get("schema") != ANSWER_SCHEMA:
        failure_reasons.append("schema_mismatch")
    if status not in ANSWERED_STATUSES and status not in ABSTENTION_STATUSES:
        failure_reasons.append("unknown_answer_status")

    citation_validation = validate_citation_bundle(citations, index) if citations else {
        "schema": "retrieval_citation_validation_v1",
        "valid": False,
        "citation_count": 0,
        "results": [],
        "failure_reason": "missing_or_invalid_citations",
        "quality_claim": "none",
        "proves_truth": False,
    }
    if status in ANSWERED_STATUSES:
        if not answer_text.strip():
            failure_reasons.append("answered_status_requires_answer_text")
        if not citations:
            failure_reasons.append("answered_status_requires_citations")
        if not citation_validation.get("valid"):
            failure_reasons.append("invalid_citations")
        if not _citation_answer_markers_present(answer_text, citations):
            failure_reasons.append("answer_text_missing_citation_marker")
    elif citations:
        failure_reasons.append("abstention_status_must_not_include_citations")

    support = claim_support_report(
        answer_text=answer_text,
        citations=citations,
        index=index,
        citation_validation=citation_validation,
    )
    if status in ANSWERED_STATUSES:
        if support.get("claim_count", 0) <= 0:
            failure_reasons.append("answered_status_requires_checkable_claim")
        if support.get("claims_unsupported", 0) > 0:
            failure_reasons.append("unsupported_claims_present")
        if support.get("claims_missing_citations", 0) > 0:
            failure_reasons.append("missing_claim_citations_present")
        if support.get("claims_contradicted", 0) > 0:
            failure_reasons.append("contradicted_claims_present")
        if status == "answered_with_citations" and support.get("claims_partially_supported", 0) > 0:
            failure_reasons.append("weak_claim_support_present")
        if status == "partial_answer_with_caveats":
            if support.get("claims_supported", 0) + support.get("claims_partially_supported", 0) <= 0:
                failure_reasons.append("partial_answer_requires_some_supported_claim")

    source_policy_status = "allowed"
    if citation_validation.get("results") and any(
        item.get("source_policy_status") == "blocked" for item in citation_validation.get("results", [])
    ):
        source_policy_status = "blocked"
        failure_reasons.append("source_policy_blocked")
    elif status in {"unsupported_current_phase", "abstained_out_of_scope", "blocked_source_policy"}:
        source_policy_status = str(answer_payload.get("source_policy_status") or "out_of_scope")

    valid_citation_count = sum(1 for item in citation_validation.get("results", []) if item.get("valid"))
    resolved_citation_count = sum(
        1 for item in citation_validation.get("results", [])
        if item.get("locator_resolution_status") == "resolved"
    )
    valid = not failure_reasons and (status in ABSTENTION_STATUSES or bool(citation_validation.get("valid")))
    return {
        "schema": ANSWER_VALIDATION_SCHEMA,
        "valid": valid,
        "answer_verifier_valid": valid,
        "failure_reasons": sorted(set(failure_reasons)),
        "failure_reason": None if valid else ",".join(sorted(set(failure_reasons))),
        "answer_status": status,
        "citation_count": len(citations),
        "valid_citation_count": valid_citation_count,
        "resolved_citation_count": resolved_citation_count,
        "citation_validation": citation_validation,
        "support_level": "lexical_or_locator_only" if citations else "none",
        "claim_count": support.get("claim_count", 0),
        "claims_checked": support.get("claims_checked", 0),
        "claims_supported": support.get("claims_supported", 0),
        "claims_partially_supported": support.get("claims_partially_supported", 0),
        "claims_unsupported": support.get("claims_unsupported", 0),
        "claims_contradicted": support.get("claims_contradicted", 0),
        "claims_missing_citations": support.get("claims_missing_citations", 0),
        "contradiction_detected": bool(support.get("contradiction_detected")),
        "answer_support_quality": support.get("answer_support_quality"),
        "claim_support_level": support.get("claim_support_level"),
        "unsupported_claims": support.get("unsupported_claims", []),
        "partially_supported_claims": support.get("partially_supported_claims", []),
        "contradicted_claims": support.get("contradicted_claims", []),
        "missing_citation_claims": support.get("missing_citation_claims", []),
        "claim_support_report": support,
        "semantic_support_level": support.get("semantic_support_level"),
        "semantic_truth_claim": "limited_or_none",
        "source_policy_status": source_policy_status,
        "claim_support_check": support,
        "quality_claim": "none",
        "proves_truth": False,
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
