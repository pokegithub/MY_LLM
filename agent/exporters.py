"""Truthful trajectory exporters for future learning workflows."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agent.trajectory import trajectory_root
from config import agent_cfg


EXPORT_REPORT_SCHEMA = "agent_export_report_v1"
SFT_EXPORT_SCHEMA = "agent_sft_export_v1"
PREFERENCE_EXPORT_SCHEMA = "agent_preference_export_v1"
RETRIEVAL_EXPORT_SCHEMA = "agent_retrieval_export_v1"
EXPORTER_VERSION = "phase4_v1"


def export_root() -> str:
    report_dir = Path(agent_cfg.report_dir).resolve()
    return str((report_dir.parent / "agent_exports").resolve())


def _now_ts() -> int:
    return int(time.time())


def _new_export_dir(export_type: str) -> tuple[str, str]:
    run_id = f"{export_type}_{_now_ts()}_{uuid.uuid4().hex[:8]}"
    out_dir = os.path.join(export_root(), run_id)
    os.makedirs(out_dir, exist_ok=True)
    return run_id, out_dir


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _write_jsonl(path: str, records: Sequence[Mapping[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


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


def _load_trajectories(
    *,
    root: Optional[str] = None,
    run_id: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _iter_trajectory_paths(root=root):
        record = _load_trajectory(path)
        if run_id and str(record.get("run_id")) != run_id:
            continue
        if route and str(record.get("route")) != route:
            continue
        if status and str(record.get("status")) != status:
            continue
        records.append(record)
    records.sort(
        key=lambda item: (int(item.get("timestamp", 0)), str(item.get("run_id", ""))),
        reverse=True,
    )
    return records


def _shorten(text: str, max_chars: int = 240) -> str:
    value = str(text or "").strip().replace("\n", " ")
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _basename_list(paths: Sequence[str]) -> List[str]:
    return [os.path.basename(str(path)) for path in (paths or [])]


def _first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _winning_attempt(record: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    attempts = list(record.get("attempts", []) or [])
    winning_attempt = record.get("winning_attempt")
    if isinstance(winning_attempt, int):
        for attempt in attempts:
            if int(attempt.get("attempt_index", -1)) == winning_attempt:
                return attempt
    for attempt in reversed(attempts):
        verification = attempt.get("verification") or {}
        if attempt.get("kept") and bool(verification.get("overall_passed")):
            return attempt
    return None


def _patch_summary(record: Mapping[str, Any]) -> Optional[str]:
    winner = _winning_attempt(record)
    if winner:
        return _first_nonempty(winner.get("candidate_summary"), winner.get("candidate_id"))
    final_origin = str(record.get("final_origin", "none"))
    if final_origin == "deterministic_exact_tool":
        return "deterministic exact-tool result verified"
    if final_origin == "verification_only":
        return "workspace verification passed without candidate generation"
    return None


def _lesson_summary(record: Mapping[str, Any]) -> Optional[str]:
    lessons = list(record.get("lessons", []) or [])
    if lessons:
        return _shorten(lessons[0], max_chars=220)
    return _first_nonempty(
        record.get("verification_summary"),
        record.get("blocked_reason"),
    )


def _provenance(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_run_id": record.get("run_id"),
        "source_status": record.get("status"),
        "source_final_origin": record.get("final_origin", "none"),
        "source_operation": record.get("operation"),
        "source_quality_claim": record.get("quality_claim", "none"),
        "source_trajectory_path": record.get("_trajectory_path", ""),
    }


def _build_export_report(
    *,
    export_type: str,
    record_schema: str,
    out_dir: str,
    records: Sequence[Mapping[str, Any]],
    scanned: int,
    skipped: int,
    rejected: int,
    skip_reasons: Mapping[str, int],
    reject_reasons: Mapping[str, int],
    root: str,
) -> Dict[str, Any]:
    records_path = os.path.join(out_dir, "records.jsonl")
    report_path = os.path.join(out_dir, "report.json")
    _write_jsonl(records_path, records)
    payload = {
        "schema": EXPORT_REPORT_SCHEMA,
        "export_type": export_type,
        "record_schema": record_schema,
        "exporter_version": EXPORTER_VERSION,
        "timestamp": _now_ts(),
        "trajectory_root": os.path.abspath(root),
        "output_dir": os.path.abspath(out_dir),
        "records_path": os.path.abspath(records_path),
        "report_path": os.path.abspath(report_path),
        "scanned_trajectories": scanned,
        "exported_records": len(records),
        "skipped_trajectories": skipped,
        "rejected_trajectories": rejected,
        "skip_reason_counts": dict(skip_reasons),
        "reject_reason_counts": dict(reject_reasons),
        "useful_output": len(records) > 0,
        "preview": list(records[:3]),
        "learning_claim": "none",
        "model_improvement_claim": "none",
        "intended_use_notice": "training/use artifacts only; no model improvement is claimed by this export",
    }
    _write_json(report_path, payload)
    return payload


def _increment(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _validate_common_provenance(record: Mapping[str, Any]) -> Optional[str]:
    required = {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "operation": record.get("operation"),
        "_trajectory_path": record.get("_trajectory_path"),
        "task_text": record.get("task_text"),
        "route": record.get("route"),
    }
    missing = [key for key, value in required.items() if not str(value or "").strip()]
    if missing:
        return "missing_required_provenance:" + ",".join(sorted(missing))
    return None


def export_sft_records(
    *,
    root: Optional[str] = None,
    run_id: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    store_root = root or trajectory_root()
    export_run_id, out_dir = _new_export_dir("trajectory_export_sft")
    scanned = 0
    skipped = 0
    rejected = 0
    skip_reasons: Dict[str, int] = {}
    reject_reasons: Dict[str, int] = {}
    exported: List[Dict[str, Any]] = []

    for record in _load_trajectories(root=store_root, run_id=run_id, route=route, status=status):
        scanned += 1
        provenance_error = _validate_common_provenance(record)
        if provenance_error:
            rejected += 1
            _increment(reject_reasons, provenance_error)
            continue
        if str(record.get("status")) != "verified_success":
            skipped += 1
            _increment(skip_reasons, "status_not_verified_success")
            continue
        if str(record.get("operation")) != "solve":
            skipped += 1
            _increment(skip_reasons, "operation_not_solve")
            continue
        if str(record.get("quality_claim")) != "verification_passed":
            rejected += 1
            _increment(reject_reasons, "verified_success_without_verifier_passed_quality_claim")
            continue
        outcome_summary = _first_nonempty(record.get("verification_summary"), _lesson_summary(record))
        if not outcome_summary:
            rejected += 1
            _increment(reject_reasons, "missing_verified_outcome_summary")
            continue
        exported.append(
            {
                "schema": SFT_EXPORT_SCHEMA,
                "export_timestamp": _now_ts(),
                "exporter_version": EXPORTER_VERSION,
                "learning_claim": "none",
                "model_improvement_claim": "none",
                "intended_use": "future_supervised_training_input_only",
                "evidence_label": "verified_trajectory_verifier_passed",
                "provenance": _provenance(record),
                "prompt": {
                    "task_text": record.get("task_text", ""),
                    "route": record.get("route"),
                    "context_summary": record.get("context_summary", {}),
                    "plan_summary": record.get("plan_summary", {}),
                },
                "target": {
                    "final_status": record.get("status"),
                    "final_origin": record.get("final_origin", "none"),
                    "verified_outcome_summary": outcome_summary,
                    "patch_summary": _patch_summary(record),
                    "lesson_summary": _lesson_summary(record),
                    "files_touched": list(record.get("files_touched", []) or []),
                },
            }
        )

    return _build_export_report(
        export_type=export_run_id,
        record_schema=SFT_EXPORT_SCHEMA,
        out_dir=out_dir,
        records=exported,
        scanned=scanned,
        skipped=skipped,
        rejected=rejected,
        skip_reasons=skip_reasons,
        reject_reasons=reject_reasons,
        root=store_root,
    )


def _build_preference_pair(
    *,
    source_record: Mapping[str, Any],
    winner: Mapping[str, Any],
    loser: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    winner_verification = winner.get("verification") or {}
    loser_verification = loser.get("verification") or {}
    if not bool(winner_verification.get("overall_passed")):
        return None
    if bool(loser_verification.get("overall_passed")):
        return None
    if not bool(loser_verification.get("meaningful", True)):
        return None

    loser_critique = loser.get("critique") or {}
    loser_failure_class = loser_critique.get("failure_class")
    loser_summary = _first_nonempty(
        loser_verification.get("summary"),
        loser_critique.get("evidence_summary"),
    )
    winner_summary = _first_nonempty(
        winner_verification.get("summary"),
        winner.get("candidate_summary"),
    )
    if not loser_summary or not winner_summary:
        return None

    pair_kind = "verified_candidate_over_failed_attempt"
    if str(loser.get("phase")) == "optimization":
        pair_kind = "verified_candidate_over_regressing_optimization"
    elif str(winner.get("phase")) == "repair" and str(loser.get("phase")) == "initial":
        pair_kind = "repaired_candidate_over_failed_initial"

    ranking_reason = "winner passed verification while loser failed verification"
    if loser_failure_class:
        ranking_reason += f" with {loser_failure_class}"

    return {
        "schema": PREFERENCE_EXPORT_SCHEMA,
        "export_timestamp": _now_ts(),
        "exporter_version": EXPORTER_VERSION,
        "learning_claim": "none",
        "model_improvement_claim": "none",
        "intended_use": "future_preference_optimization_input_only",
        "evidence_label": "verifier_ranked_pair",
        "pair_kind": pair_kind,
        "provenance": _provenance(source_record),
        "task_text": source_record.get("task_text", ""),
        "route": source_record.get("route"),
        "winner": {
            "attempt_index": winner.get("attempt_index"),
            "phase": winner.get("phase"),
            "origin": winner.get("origin"),
            "candidate_id": winner.get("candidate_id"),
            "candidate_summary": winner.get("candidate_summary"),
            "files_touched": list(winner.get("files_touched", []) or []),
            "verification_summary": winner_summary,
        },
        "loser": {
            "attempt_index": loser.get("attempt_index"),
            "phase": loser.get("phase"),
            "origin": loser.get("origin"),
            "candidate_id": loser.get("candidate_id"),
            "candidate_summary": loser.get("candidate_summary"),
            "files_touched": list(loser.get("files_touched", []) or []),
            "verification_summary": loser_summary,
            "failure_class": loser_failure_class,
        },
        "ranking_reason": ranking_reason,
        "verifier_justification": {
            "winner_overall_passed": True,
            "loser_overall_passed": False,
            "loser_failure_class": loser_failure_class,
        },
    }


def export_preference_records(
    *,
    root: Optional[str] = None,
    run_id: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    store_root = root or trajectory_root()
    export_run_id, out_dir = _new_export_dir("trajectory_export_preferences")
    scanned = 0
    skipped = 0
    rejected = 0
    skip_reasons: Dict[str, int] = {}
    reject_reasons: Dict[str, int] = {}
    exported: List[Dict[str, Any]] = []

    for record in _load_trajectories(root=store_root, run_id=run_id, route=route, status=status):
        scanned += 1
        provenance_error = _validate_common_provenance(record)
        if provenance_error:
            rejected += 1
            _increment(reject_reasons, provenance_error)
            continue
        if str(record.get("status")) != "verified_success":
            skipped += 1
            _increment(skip_reasons, "status_not_verified_success")
            continue
        if str(record.get("operation")) != "solve":
            skipped += 1
            _increment(skip_reasons, "operation_not_solve")
            continue
        if str(record.get("quality_claim")) != "verification_passed":
            rejected += 1
            _increment(reject_reasons, "verified_success_without_verifier_passed_quality_claim")
            continue
        winner = _winning_attempt(record)
        if winner is None:
            skipped += 1
            _increment(skip_reasons, "no_verifier_justified_preference_pair")
            continue
        attempts = list(record.get("attempts", []) or [])
        pairs: List[Dict[str, Any]] = []
        for loser in attempts:
            if loser is winner:
                continue
            if loser.get("kept"):
                continue
            pair = _build_preference_pair(
                source_record=record,
                winner=winner,
                loser=loser,
            )
            if pair is not None:
                pairs.append(pair)
        if not pairs:
            skipped += 1
            _increment(skip_reasons, "no_verifier_justified_preference_pair")
            continue
        exported.extend(pairs)

    return _build_export_report(
        export_type=export_run_id,
        record_schema=PREFERENCE_EXPORT_SCHEMA,
        out_dir=out_dir,
        records=exported,
        scanned=scanned,
        skipped=skipped,
        rejected=rejected,
        skip_reasons=skip_reasons,
        reject_reasons=reject_reasons,
        root=store_root,
    )


def export_retrieval_records(
    *,
    root: Optional[str] = None,
    run_id: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    store_root = root or trajectory_root()
    export_run_id, out_dir = _new_export_dir("trajectory_export_retrieval")
    scanned = 0
    skipped = 0
    rejected = 0
    skip_reasons: Dict[str, int] = {}
    reject_reasons: Dict[str, int] = {}
    exported: List[Dict[str, Any]] = []

    status_to_type = {
        "verified_success": "verified_solution",
        "verification_failed": "failure_pattern",
        "blocked_unverified": "blocked_attempt",
        "unsupported": "unsupported_route",
        "plan_ready": "planning_only",
    }
    status_to_evidence = {
        "verified_success": "verifier_passed",
        "verification_failed": "verifier_failed",
        "blocked_unverified": "blocked_without_verifier_pass",
        "unsupported": "unsupported_route",
        "plan_ready": "planning_only_no_verification",
    }

    for record in _load_trajectories(root=store_root, run_id=run_id, route=route, status=status):
        scanned += 1
        provenance_error = _validate_common_provenance(record)
        if provenance_error:
            rejected += 1
            _increment(reject_reasons, provenance_error)
            continue
        trajectory_status = str(record.get("status"))
        if trajectory_status not in status_to_type:
            rejected += 1
            _increment(reject_reasons, "unknown_trajectory_status")
            continue
        task_text = str(record.get("task_text", "")).strip()
        if not task_text:
            rejected += 1
            _increment(reject_reasons, "missing_task_text")
            continue
        fix_summary = _patch_summary(record)
        lesson_summary = _lesson_summary(record)
        retrieval_text = (
            f"task={_shorten(task_text, 140)}; "
            f"status={trajectory_status}; "
            f"route={record.get('route')}; "
            f"files={','.join(_basename_list(record.get('files_touched', []) or [])) or 'none'}; "
            f"fix={_shorten(fix_summary or 'none', 100)}; "
            f"lesson={_shorten(lesson_summary or 'none', 120)}"
        )
        exported.append(
            {
                "schema": RETRIEVAL_EXPORT_SCHEMA,
                "export_timestamp": _now_ts(),
                "exporter_version": EXPORTER_VERSION,
                "learning_claim": "none",
                "model_improvement_claim": "none",
                "intended_use": "future_retrieval_or_memory_input_only",
                "evidence_label": status_to_evidence[trajectory_status],
                "experience_type": status_to_type[trajectory_status],
                "provenance": _provenance(record),
                "task_summary": _shorten(task_text, 180),
                "route": record.get("route"),
                "touched_files": list(record.get("files_touched", []) or []),
                "failure_classes": list(record.get("failure_classes", []) or []),
                "final_fix_summary": fix_summary,
                "outcome_status": trajectory_status,
                "lesson_summary": lesson_summary,
                "retrieval_text": _shorten(retrieval_text, 320),
            }
        )

    return _build_export_report(
        export_type=export_run_id,
        record_schema=RETRIEVAL_EXPORT_SCHEMA,
        out_dir=out_dir,
        records=exported,
        scanned=scanned,
        skipped=skipped,
        rejected=rejected,
        skip_reasons=skip_reasons,
        reject_reasons=reject_reasons,
        root=store_root,
    )
