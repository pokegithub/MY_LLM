"""Deterministic trajectory retrieval for Phase 3 agent memory."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agent.trajectory import trajectory_root


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STATUS_WEIGHTS = {
    "verified_success": 300,
    "verification_failed": 200,
    "blocked_unverified": 100,
    "unsupported": 50,
}


def _tokenize(text: str) -> List[str]:
    return [match.group(0) for match in _TOKEN_RE.finditer((text or "").lower())]


def _iter_trajectory_paths(root: Optional[str] = None) -> List[str]:
    base = root or trajectory_root()
    if not os.path.isdir(base):
        return []
    paths: List[str] = []
    for child in Path(base).iterdir():
        if not child.is_dir():
            continue
        trajectory_path = child / "trajectory.json"
        if trajectory_path.is_file():
            paths.append(str(trajectory_path.resolve()))
    return sorted(paths)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _path_matches(candidate_path: str, touched_file: str) -> bool:
    lhs = str(candidate_path).replace("\\", "/").lower()
    rhs = str(touched_file).replace("\\", "/").lower()
    return rhs in lhs or os.path.basename(lhs) == os.path.basename(rhs)


def _matches_filters(
    record: Mapping[str, Any],
    *,
    route: Optional[str] = None,
    status: Optional[str] = None,
    touched_file: Optional[str] = None,
    failure_class: Optional[str] = None,
    run_id: Optional[str] = None,
) -> bool:
    if route and str(record.get("route")) != route:
        return False
    if status and str(record.get("status")) != status:
        return False
    if run_id and str(record.get("run_id")) != run_id:
        return False
    if touched_file:
        touched = record.get("files_touched", []) or []
        if not any(_path_matches(str(item), touched_file) for item in touched):
            return False
    if failure_class:
        values = record.get("failure_classes", []) or []
        if failure_class not in values:
            return False
    return True


def _summary(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": record.get("run_id"),
        "timestamp": record.get("timestamp"),
        "operation": record.get("operation"),
        "route": record.get("route"),
        "status": record.get("status"),
        "final_origin": record.get("final_origin", "none"),
        "optimization_status": record.get("optimization_status", "not_attempted"),
        "task_text": record.get("task_text", ""),
        "files_touched": list(record.get("files_touched", []) or []),
        "failure_classes": list(record.get("failure_classes", []) or []),
        "trajectory_path": record.get("_trajectory_path", record.get("artifact_paths", {}).get("agent_report")),
        "blocked_reason": record.get("blocked_reason"),
        "quality_claim": record.get("quality_claim", "none"),
    }


def list_trajectories(
    *,
    root: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
    touched_file: Optional[str] = None,
    failure_class: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _iter_trajectory_paths(root=root):
        record = _load_json(path)
        if _matches_filters(
            record,
            route=route,
            status=status,
            touched_file=touched_file,
            failure_class=failure_class,
        ):
            record["_trajectory_path"] = path
            records.append(_summary(record))
    records.sort(key=lambda item: (int(item.get("timestamp", 0)), str(item.get("run_id", ""))), reverse=True)
    return records[: max(0, int(limit))]


def show_trajectory(run_id: str, *, root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for path in _iter_trajectory_paths(root=root):
        record = _load_json(path)
        if str(record.get("run_id")) == run_id:
            record["_trajectory_path"] = path
            return record
    return None


def _search_score(record: Mapping[str, Any], *, query_tokens: Sequence[str], file_tokens: Sequence[str]) -> int:
    score = _STATUS_WEIGHTS.get(str(record.get("status")), 0)
    task_tokens = set(_tokenize(str(record.get("task_text", ""))))
    route = str(record.get("route", ""))
    touched = " ".join(str(item) for item in (record.get("files_touched", []) or []))
    touched_tokens = set(_tokenize(touched))

    for token in query_tokens:
        if token in task_tokens:
            score += 15
    for token in file_tokens:
        if token in touched_tokens:
            score += 20
    if route == "coding_edit":
        score += 5
    return score


def search_trajectories(
    *,
    query: str,
    root: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
    touched_file: Optional[str] = None,
    failure_class: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(query)
    file_tokens = _tokenize(touched_file or "")
    ranked: List[Dict[str, Any]] = []

    for record in list_trajectories(
        root=root,
        route=route,
        status=status,
        touched_file=touched_file,
        failure_class=failure_class,
        limit=10_000,
    ):
        score = _search_score(record, query_tokens=query_tokens, file_tokens=file_tokens)
        if score <= 0:
            continue
        record = dict(record)
        record["_search_score"] = score
        ranked.append(record)

    ranked.sort(
        key=lambda item: (
            int(item.get("_search_score", 0)),
            int(item.get("timestamp", 0)),
            str(item.get("run_id", "")),
        ),
        reverse=True,
    )
    return ranked[: max(0, int(limit))]


def summarize_for_context(
    *,
    query: str,
    file_hints: Sequence[str],
    root: Optional[str] = None,
    limit: int = 2,
    max_chars: int = 220,
) -> List[str]:
    results = search_trajectories(
        query=query,
        root=root,
        touched_file=file_hints[0] if file_hints else None,
        limit=max(0, int(limit)) * 3,
    )
    summaries: List[str] = []
    for record in results:
        status = str(record.get("status"))
        if status not in {"verified_success", "verification_failed"}:
            continue
        task_text = str(record.get("task_text", "")).strip().replace("\n", " ")
        failure_classes = record.get("failure_classes", []) or []
        touched = record.get("files_touched", []) or []
        note = (
            f"prior trajectory {record.get('run_id')}: status={status}; "
            f"files={', '.join(os.path.basename(str(item)) for item in touched) or 'none'}; "
            f"failures={','.join(failure_classes) or 'none'}; "
            f"task={task_text}"
        )
        if len(note) > max_chars:
            note = note[: max_chars - 3] + "..."
        summaries.append(note)
        if len(summaries) >= limit:
            break
    if not summaries:
        return ["No prior similar verified or failed coding trajectories were found."]
    return summaries
