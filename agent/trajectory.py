"""Replayable trajectory storage for verified coding-agent runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from config import agent_cfg


TRAJECTORY_SCHEMA = "agent_trajectory_v1"


def trajectory_root() -> str:
    """Return the persistent trajectory store root."""
    report_dir = Path(agent_cfg.report_dir).resolve()
    return str((report_dir.parent / "agent_trajectories").resolve())


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _summarize_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    files = context.get("files", []) or []
    included = []
    for item in files:
        if not item.get("included"):
            continue
        included.append(
            {
                "path": item.get("path"),
                "note": item.get("note", ""),
                "size_bytes": item.get("size_bytes", 0),
            }
        )
    return {
        "workspace_root": context.get("workspace_root"),
        "files_considered": list(context.get("files_considered", []) or []),
        "included_files": included,
        "notes": list(context.get("notes", []) or []),
    }


def _summarize_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "route": plan.get("route"),
        "subgoals": list(plan.get("subgoals", []) or []),
        "target_files": list(plan.get("target_files", []) or []),
        "checks": list(plan.get("checks", []) or []),
        "risk_level": plan.get("risk_level"),
        "success_criteria": list(plan.get("success_criteria", []) or []),
        "retry_budget": plan.get("retry_budget", 0),
        "stop_conditions": list(plan.get("stop_conditions", []) or []),
    }


def _summarize_attempts(attempts: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for attempt in attempts:
        critique = attempt.get("critique") or None
        summaries.append(
            {
                "attempt_index": attempt.get("attempt_index"),
                "phase": attempt.get("phase"),
                "origin": attempt.get("origin"),
                "candidate_id": attempt.get("candidate_id"),
                "candidate_summary": attempt.get("candidate_summary"),
                "files_touched": list(attempt.get("files_touched", []) or []),
                "kept": bool(attempt.get("kept", False)),
                "verification": attempt.get("verification"),
                "critique": critique,
            }
        )
    return summaries


def _failure_classes(attempts: List[Mapping[str, Any]]) -> List[str]:
    values = []
    for attempt in attempts:
        critique = attempt.get("critique") or {}
        failure_class = critique.get("failure_class")
        if failure_class and failure_class not in values:
            values.append(str(failure_class))
    return values


def _build_lessons(report: Mapping[str, Any], attempts: List[Mapping[str, Any]]) -> List[str]:
    status = str(report.get("status", "unknown"))
    final_origin = str(report.get("final_origin", "none"))
    optimization_status = str(report.get("optimization_status", "not_attempted"))
    lessons: List[str] = []

    if status == "verified_success":
        lessons.append(f"Verification passed with final_origin={final_origin}.")
    elif status == "verification_failed":
        lessons.append("All available attempts failed verification within the configured retry budget.")
    elif status == "blocked_unverified":
        blocked_reason = report.get("blocked_reason") or "Solve path was blocked without verification."
        lessons.append(str(blocked_reason))
    elif status == "unsupported":
        lessons.append("Task route remained unsupported in the current agent phase.")

    failure_classes = _failure_classes(attempts)
    if failure_classes:
        lessons.append("Observed failure classes: " + ", ".join(failure_classes))
    if optimization_status == "attempted_and_rolled_back":
        lessons.append("A post-green optimization regressed verification and was rolled back.")
    elif optimization_status == "attempted_and_kept":
        lessons.append("A post-green optimization kept verification green.")
    return lessons


def build_trajectory_record(report: Mapping[str, Any]) -> Dict[str, Any]:
    attempts = list(report.get("attempts", []) or [])
    request = dict(report.get("request", {}) or {})
    context = dict(report.get("context", {}) or {})
    plan = dict(report.get("plan", {}) or {})
    verification = dict(report.get("verification", {}) or {})
    backend_status = dict(report.get("backend_status", {}) or {})
    route = dict(report.get("route", {}) or {})
    report_path = str(report.get("report_path", ""))
    touched_files = list(report.get("files_touched", []) or [])
    failure_classes = _failure_classes(attempts)

    return {
        "schema": TRAJECTORY_SCHEMA,
        "run_id": report.get("run_id"),
        "timestamp": int(report.get("timestamp", time.time())),
        "operation": report.get("operation"),
        "task_text": request.get("task_text", ""),
        "request": {
            "task_text": request.get("task_text", ""),
            "file_hints": list(request.get("file_hints", []) or []),
            "checks": list(request.get("checks", []) or []),
            "task_file": request.get("task_file"),
            "workspace_root": request.get("workspace_root"),
            "task_type_hint": request.get("task_type_hint"),
        },
        "route": route.get("route"),
        "route_reason": route.get("reason"),
        "backend_status": backend_status,
        "context_summary": _summarize_context(context),
        "plan_summary": _summarize_plan(plan),
        "files_considered": list(context.get("files_considered", []) or []),
        "files_touched": touched_files,
        "retry_budget": int(report.get("retry_budget", 0) or 0),
        "attempt_count": len(attempts),
        "attempts": _summarize_attempts(attempts),
        "failure_classes": failure_classes,
        "status": report.get("status"),
        "final_origin": report.get("final_origin", "none"),
        "winning_attempt": report.get("winning_attempt"),
        "optimization_status": report.get("optimization_status", "not_attempted"),
        "blocked_reason": report.get("blocked_reason"),
        "quality_claim": report.get("quality_claim", "none"),
        "verification_summary": verification.get("summary", ""),
        "lessons": _build_lessons(report, attempts),
        "artifact_paths": {
            "agent_report": os.path.abspath(report_path) if report_path else "",
        },
        "degraded_mode": bool(report.get("degraded_mode", False)),
        "artifact_warnings": list(report.get("artifact_warnings", []) or []),
        "learning_claim": "none",
    }


def write_trajectory_record(report: Mapping[str, Any]) -> Dict[str, Any]:
    record = build_trajectory_record(report)
    root = trajectory_root()
    run_id = str(record["run_id"])
    run_dir = os.path.join(root, run_id)
    path = os.path.join(run_dir, "trajectory.json")
    _write_json(path, record)
    return {
        "ok": True,
        "required": False,
        "non_critical": True,
        "schema": TRAJECTORY_SCHEMA,
        "path": os.path.abspath(path),
        "root": os.path.abspath(root),
    }
