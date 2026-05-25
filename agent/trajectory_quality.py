"""Deterministic trajectory-quality audit for future training-use filtering."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agent.trajectory import trajectory_root
from config import agent_cfg


QUALITY_AUDIT_SCHEMA = "trajectory_quality_audit_v1"
QUALITY_CLASS_SCHEMA = "trajectory_quality_schema_v1"
QUALITY_AUDITOR_VERSION = "phase_a_trajectory_quality_v1"

SFT_POSITIVE = "sft_positive_eligible"
PREFERENCE_WINNER = "preference_winner_eligible"
PREFERENCE_LOSER = "preference_loser_eligible"
RETRIEVAL_MEMORY = "retrieval_memory_eligible"
FAILURE_ANALYSIS = "failure_analysis_eligible"
EXACT_TOOL_ONLY = "exact_tool_curriculum_only"
FIXTURE_ONLY = "fixture_only"
BACKENDLESS_BLOCKED = "backendless_blocked"
BLOCKED_UNVERIFIED = "blocked_unverified"
INSUFFICIENT_PROVENANCE = "insufficient_provenance"
REJECTED_LOW_VALUE = "rejected_low_value"
UNKNOWN_UNCLASSIFIED = "unknown_unclassified"

REAL_BACKEND_KINDS = {
    "local_transformers_in_process",
}
NON_REAL_BACKEND_KINDS = {
    "",
    "none",
    "scripted",
    "deterministic_exact_tool",
}
FIXTURE_MARKERS = (
    "run_artifacts/test_agent_",
    "run_artifacts\\test_agent_",
    "test_agent_phase",
    "test_agent_phase1_workspaces",
    "test_agent_phase2_workspaces",
    "test_agent_phase3_workspaces",
    "test_agent_phase4_workspaces",
    "run_artifacts/qwen_tiny_solve_workspace",
    "run_artifacts/hidden_eval_runs",
    "backend_smoke_target.py",
    "candidate_readiness_",
)


def quality_report_root() -> str:
    report_dir = Path(agent_cfg.report_dir).resolve()
    return str((report_dir.parent / "trajectory_quality").resolve())


def _iter_trajectory_paths(root: Optional[str] = None) -> List[str]:
    base = root or trajectory_root()
    if not os.path.isdir(base):
        return []
    paths: List[str] = []
    for child in Path(base).iterdir():
        if not child.is_dir():
            continue
        path = child / "trajectory.json"
        if path.is_file():
            paths.append(str(path.resolve()))
    return sorted(paths)


def _load_trajectory(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_trajectory_path"] = path
    return payload


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return os.path.abspath(path)


def _new_report_dir() -> str:
    run_id = f"trajectory_quality_audit_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    out_dir = os.path.join(quality_report_root(), run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _all_text_fields(record: Mapping[str, Any]) -> str:
    chunks = [
        record.get("task_text", ""),
        record.get("run_id", ""),
        record.get("route", ""),
        record.get("final_origin", ""),
        record.get("blocked_reason", ""),
        record.get("_trajectory_path", ""),
    ]
    for key in ("files_touched", "files_considered"):
        chunks.extend(str(item) for item in _as_list(record.get(key)))
    context = record.get("context_summary") or {}
    if isinstance(context, Mapping):
        chunks.append(str(context.get("workspace_root", "")))
        chunks.extend(str(item) for item in _as_list(context.get("files_considered")))
        for item in _as_list(context.get("included_files")):
            if isinstance(item, Mapping):
                chunks.append(str(item.get("path", "")))
    return "\n".join(str(item) for item in chunks).replace("\\", "/").lower()


def _is_fixture_trace(record: Mapping[str, Any]) -> bool:
    text = _all_text_fields(record)
    return any(marker.replace("\\", "/").lower() in text for marker in FIXTURE_MARKERS)


def _backend_kind(record: Mapping[str, Any]) -> str:
    backend = record.get("backend_status") or {}
    if not isinstance(backend, Mapping):
        return ""
    return str(backend.get("kind", "")).strip().lower()


def _is_real_backend(record: Mapping[str, Any]) -> bool:
    backend = record.get("backend_status") or {}
    if not isinstance(backend, Mapping):
        return False
    return (
        str(backend.get("kind", "")).strip().lower() in REAL_BACKEND_KINDS
        and bool(backend.get("configured"))
        and bool(backend.get("available"))
    )


def _attempts(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [item for item in _as_list(record.get("attempts")) if isinstance(item, Mapping)]


def _attempt_verification(attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    verification = attempt.get("verification") or {}
    if isinstance(verification, Mapping):
        return verification
    return {}


def _has_meaningful_verifier_pass(record: Mapping[str, Any]) -> bool:
    if str(record.get("quality_claim")) != "verification_passed":
        return False
    if str(record.get("status")) != "verified_success":
        return False
    attempts = _attempts(record)
    if not attempts:
        return str(record.get("verification_summary", "")).strip() == "all verification checks passed"
    for attempt in attempts:
        verification = _attempt_verification(attempt)
        if bool(verification.get("overall_passed")) and bool(verification.get("meaningful", True)):
            return True
    return False


def _has_failed_attempt_evidence(record: Mapping[str, Any]) -> bool:
    for attempt in _attempts(record):
        verification = _attempt_verification(attempt)
        if verification and not bool(verification.get("overall_passed")) and bool(verification.get("meaningful", True)):
            return True
    return bool(record.get("failure_classes"))


def _has_preference_pair_evidence(record: Mapping[str, Any]) -> bool:
    has_winner = False
    has_loser = False
    for attempt in _attempts(record):
        verification = _attempt_verification(attempt)
        if bool(attempt.get("kept")) and bool(verification.get("overall_passed")):
            has_winner = True
        if not bool(verification.get("overall_passed")) and bool(verification.get("meaningful", True)):
            has_loser = True
    return has_winner and has_loser


def _missing_required_fields(record: Mapping[str, Any]) -> List[str]:
    required = {
        "run_id": record.get("run_id"),
        "operation": record.get("operation"),
        "task_text": record.get("task_text"),
        "route": record.get("route"),
        "status": record.get("status"),
    }
    return sorted(key for key, value in required.items() if not _nonempty(value))


def classify_trajectory(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify one stored trajectory for future-use eligibility."""
    run_id = str(record.get("run_id", ""))
    route = str(record.get("route", ""))
    status = str(record.get("status", ""))
    operation = str(record.get("operation", ""))
    backend_kind = _backend_kind(record)
    missing = _missing_required_fields(record)
    classes: List[str] = []
    reasons: List[str] = []

    if missing:
        classes.extend([INSUFFICIENT_PROVENANCE, REJECTED_LOW_VALUE])
        reasons.append("missing_required_provenance:" + ",".join(missing))
    else:
        is_fixture = _is_fixture_trace(record)
        if is_fixture:
            classes.append(FIXTURE_ONLY)
            reasons.append("fixture_or_test_workspace_marker_detected")

        if route == "exact_symbolic" and status == "verified_success" and str(record.get("final_origin")) == "deterministic_exact_tool":
            classes.extend([EXACT_TOOL_ONLY, RETRIEVAL_MEMORY])
            reasons.append("deterministic_exact_tool_trace_not_general_coding_sft")

        if route == "coding_edit" and status == "blocked_unverified":
            if backend_kind == "none":
                classes.append(BACKENDLESS_BLOCKED)
                reasons.append("no_coding_backend_configured")
            classes.extend([BLOCKED_UNVERIFIED, FAILURE_ANALYSIS, RETRIEVAL_MEMORY])
            if record.get("blocked_reason"):
                reasons.append("blocked_reason_present")

        if status == "verification_failed":
            if _has_failed_attempt_evidence(record):
                classes.extend([PREFERENCE_LOSER, FAILURE_ANALYSIS])
                reasons.append("meaningful_verifier_failure_evidence")
            else:
                classes.append(REJECTED_LOW_VALUE)
                reasons.append("verification_failed_without_attempt_failure_evidence")
            if not is_fixture:
                classes.append(RETRIEVAL_MEMORY)

        if route == "coding_edit" and status == "verified_success":
            if backend_kind in NON_REAL_BACKEND_KINDS:
                classes.append(REJECTED_LOW_VALUE)
                reasons.append(f"non_real_backend_kind:{backend_kind or 'unknown'}")
            if is_fixture:
                classes.append(REJECTED_LOW_VALUE)
                reasons.append("fixture_trace_not_positive_training_data")
            if _is_real_backend(record) and not is_fixture and _has_meaningful_verifier_pass(record) and _as_list(record.get("files_touched")):
                classes.extend([SFT_POSITIVE, RETRIEVAL_MEMORY])
                reasons.append("real_backend_verified_coding_trace")
                if _has_preference_pair_evidence(record):
                    classes.append(PREFERENCE_WINNER)
                    reasons.append("winner_and_loser_attempts_have_verifier_evidence")
            elif not any(cls in classes for cls in (FIXTURE_ONLY, REJECTED_LOW_VALUE)):
                classes.append(REJECTED_LOW_VALUE)
                reasons.append("verified_coding_trace_missing_sft_positive_requirements")

        if operation == "plan" or status == "plan_ready":
            classes.append(REJECTED_LOW_VALUE)
            reasons.append("planning_only_not_training_data")

        if status == "unsupported":
            classes.extend([FAILURE_ANALYSIS, RETRIEVAL_MEMORY])
            reasons.append("unsupported_route_labeled_for_failure_analysis")

    if not classes:
        classes.append(UNKNOWN_UNCLASSIFIED)
        reasons.append("manual_review_required")

    deduped_classes = []
    for cls in classes:
        if cls not in deduped_classes:
            deduped_classes.append(cls)
    deduped_reasons = []
    for reason in reasons:
        if reason not in deduped_reasons:
            deduped_reasons.append(reason)

    return {
        "schema": "trajectory_quality_classification_v1",
        "run_id": run_id,
        "trajectory_path": record.get("_trajectory_path", ""),
        "task_text": record.get("task_text", ""),
        "route": route,
        "status": status,
        "operation": operation,
        "backend_kind": backend_kind,
        "primary_quality_class": deduped_classes[0],
        "quality_classes": deduped_classes,
        "eligible": {
            "sft_positive": SFT_POSITIVE in deduped_classes,
            "preference_winner": PREFERENCE_WINNER in deduped_classes,
            "preference_loser": PREFERENCE_LOSER in deduped_classes,
            "retrieval_memory": RETRIEVAL_MEMORY in deduped_classes,
            "failure_analysis": FAILURE_ANALYSIS in deduped_classes,
            "exact_tool_curriculum": EXACT_TOOL_ONLY in deduped_classes,
        },
        "rejected_for_positive_training": bool(
            set(deduped_classes)
            & {
                FIXTURE_ONLY,
                BACKENDLESS_BLOCKED,
                BLOCKED_UNVERIFIED,
                INSUFFICIENT_PROVENANCE,
                REJECTED_LOW_VALUE,
                UNKNOWN_UNCLASSIFIED,
            }
        )
        or SFT_POSITIVE not in deduped_classes,
        "reasons": deduped_reasons,
    }


def _count_eligibility(classifications: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for item in classifications if bool((item.get("eligible") or {}).get(key)))


def build_quality_audit(
    *,
    root: Optional[str] = None,
    write_report: bool = True,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    store_root = root or trajectory_root()
    classifications: List[Dict[str, Any]] = []
    load_errors: List[Dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for path in _iter_trajectory_paths(root=store_root):
        try:
            record = _load_trajectory(path)
            item = classify_trajectory(record)
        except Exception as exc:
            load_errors.append({"path": path, "error": str(exc)})
            item = {
                "schema": "trajectory_quality_classification_v1",
                "run_id": "",
                "trajectory_path": path,
                "task_text": "",
                "route": "unknown",
                "status": "unknown",
                "operation": "unknown",
                "backend_kind": "unknown",
                "primary_quality_class": INSUFFICIENT_PROVENANCE,
                "quality_classes": [INSUFFICIENT_PROVENANCE],
                "eligible": {
                    "sft_positive": False,
                    "preference_winner": False,
                    "preference_loser": False,
                    "retrieval_memory": False,
                    "failure_analysis": False,
                    "exact_tool_curriculum": False,
                },
                "rejected_for_positive_training": True,
                "reasons": ["trajectory_load_error"],
            }
        classifications.append(item)
        primary_counts.update([str(item.get("primary_quality_class"))])
        class_counts.update(str(cls) for cls in item.get("quality_classes", []) or [])
        reason_counts.update(str(reason) for reason in item.get("reasons", []) or [])
        backend_counts.update([str(item.get("backend_kind") or "unknown")])
        route_counts.update([str(item.get("route") or "unknown")])
        status_counts.update([str(item.get("status") or "unknown")])

    scanned = len(classifications)
    sft_count = _count_eligibility(classifications, "sft_positive")
    exact_count = _count_eligibility(classifications, "exact_tool_curriculum")
    rejected_count = sum(1 for item in classifications if bool(item.get("rejected_for_positive_training")))
    likely_low_value = (
        class_counts[EXACT_TOOL_ONLY]
        + class_counts[FIXTURE_ONLY]
        + class_counts[BACKENDLESS_BLOCKED]
        + backend_counts["scripted"]
    )
    store_quality_note = "unverified"
    if scanned:
        ratio = likely_low_value / max(scanned, 1)
        if ratio >= 0.5 or sft_count == 0:
            store_quality_note = "current_store_mostly_exact_fixture_stub_or_backendless_for_training_purposes"
        else:
            store_quality_note = "current_store_contains_some_training_candidates_but_requires_filtering"

    payload: Dict[str, Any] = {
        "schema": QUALITY_AUDIT_SCHEMA,
        "quality_schema": QUALITY_CLASS_SCHEMA,
        "auditor_version": QUALITY_AUDITOR_VERSION,
        "timestamp": int(time.time()),
        "trajectory_root": os.path.abspath(store_root),
        "scanned_trajectories": scanned,
        "load_error_count": len(load_errors),
        "class_counts": dict(sorted(class_counts.items())),
        "primary_class_counts": dict(sorted(primary_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "backend_kind_counts": dict(sorted(backend_counts.items())),
        "eligibility_counts": {
            "sft_positive_eligible": sft_count,
            "preference_winner_eligible": _count_eligibility(classifications, "preference_winner"),
            "preference_loser_eligible": _count_eligibility(classifications, "preference_loser"),
            "retrieval_memory_eligible": _count_eligibility(classifications, "retrieval_memory"),
            "failure_analysis_eligible": _count_eligibility(classifications, "failure_analysis"),
            "exact_tool_curriculum_only": exact_count,
            "rejected_for_positive_training": rejected_count,
        },
        "top_reject_or_skip_reasons": dict(reason_counts.most_common(12)),
        "store_quality_note": store_quality_note,
        "classifications": classifications,
        "load_errors": load_errors,
        "learning_claim": "none",
        "model_improvement_claim": "none",
        "training_readiness_claim": "filtered_audit_only",
    }
    if write_report:
        path = report_path or os.path.join(_new_report_dir(), "report.json")
        payload["report_path"] = _write_json(path, payload)
    return payload


def load_quality_schema(path: str = "./trajectory_quality/trajectory_quality_schema_v1.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
