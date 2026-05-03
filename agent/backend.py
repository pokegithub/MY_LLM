"""Backend protocol and fail-closed loaders for Phase 1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

from agent.types import Candidate, ContextBundle, FileEdit, SolvePlan, TaskRequest
from config import agent_cfg
from security.validator import get_allowed_data_roots, safe_load_json


class CodingModelBackend(Protocol):
    """Protocol for producing a single coding candidate."""

    def generate_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
    ) -> Candidate:
        """Return exactly one candidate for Phase 1."""


class ScriptedCandidateBackend:
    """Load a candidate from a machine-readable JSON file."""

    def __init__(self, script_path: str, workspace_root: str):
        allowed_roots = tuple(str(root) for root in get_allowed_data_roots()) + (workspace_root,)
        payload = safe_load_json(
            script_path,
            required_keys=("candidate_id", "summary", "edits"),
            allowed_roots=allowed_roots,
        )
        edits = []
        for item in payload["edits"]:
            edits.append(
                FileEdit(
                    path=str(item["path"]),
                    new_content=str(item["new_content"]),
                )
            )
        self._candidate = Candidate(
            candidate_id=str(payload["candidate_id"]),
            summary=str(payload["summary"]),
            edits=tuple(edits),
            source="scripted_backend",
            metadata={"script_path": str(Path(script_path).resolve())},
        )

    def generate_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
    ) -> Candidate:
        return self._candidate


def load_backend(
    *,
    workspace_root: str,
    backend_script_path: Optional[str] = None,
) -> Tuple[Optional[CodingModelBackend], Dict[str, Any]]:
    script_path = backend_script_path or agent_cfg.backend_script_path
    backend_kind = str(agent_cfg.backend_kind).strip().lower()

    if script_path:
        if not os.path.exists(script_path):
            return None, {
                "configured": True,
                "available": False,
                "kind": "scripted",
                "reason": "backend_script_path does not exist",
                "script_path": script_path,
            }
        return (
            ScriptedCandidateBackend(script_path, workspace_root=workspace_root),
            {
                "configured": True,
                "available": True,
                "kind": "scripted",
                "script_path": str(Path(script_path).resolve()),
            },
        )

    if backend_kind == "none":
        return None, {
            "configured": False,
            "available": False,
            "kind": "none",
            "reason": "no coding backend configured",
        }

    return None, {
        "configured": True,
        "available": False,
        "kind": backend_kind,
        "reason": "configured backend kind is unsupported in Phase 1",
    }
