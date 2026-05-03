"""Backend protocol and fail-closed loaders for the verified coding agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

from agent.types import AttemptRecord, Candidate, ContextBundle, Critique, FileEdit, SolvePlan, TaskRequest
from config import agent_cfg
from security.validator import get_allowed_data_roots, safe_load_json


class CodingModelBackend(Protocol):
    """Protocol for producing bounded coding candidates."""

    def generate_initial_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
    ) -> Candidate:
        """Return the first candidate for a coding task."""

    def generate_repair_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
        critique: Critique,
        previous_candidate: Candidate,
        attempt_history: Sequence[AttemptRecord],
    ) -> Optional[Candidate]:
        """Return a narrower repair candidate or None if unsupported."""

    def generate_optimization_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
        winning_candidate: Candidate,
        attempt_history: Sequence[AttemptRecord],
    ) -> Optional[Candidate]:
        """Return a post-green optimization candidate or None if unsupported."""


class ScriptedCandidateBackend:
    """Load bounded candidates from a machine-readable JSON file."""

    def __init__(self, script_path: str, workspace_root: str):
        allowed_roots = tuple(str(root) for root in get_allowed_data_roots()) + (workspace_root,)
        self._payload = safe_load_json(
            script_path,
            allowed_roots=allowed_roots,
        )
        self._script_path = str(Path(script_path).resolve())
        self._repair_cursor = 0

        if all(key in self._payload for key in ("candidate_id", "summary", "edits")):
            self._initial_candidate = self._candidate_from_payload(
                self._payload,
                source="scripted_backend_initial",
            )
            self._repair_candidates = ()
            self._optimization_candidate = None
        else:
            initial_payload = self._payload.get("initial_candidate")
            if not isinstance(initial_payload, dict):
                raise ValueError("scripted backend requires either top-level candidate fields or an initial_candidate object")
            self._initial_candidate = self._candidate_from_payload(
                initial_payload,
                source="scripted_backend_initial",
            )
            repair_payloads = self._payload.get("repair_candidates", [])
            if repair_payloads is None:
                repair_payloads = []
            if not isinstance(repair_payloads, list):
                raise ValueError("repair_candidates must be a list when provided")
            self._repair_candidates = tuple(
                self._candidate_from_payload(item, source="scripted_backend_repair")
                for item in repair_payloads
            )
            optimize_payload = self._payload.get("optimization_candidate")
            self._optimization_candidate = None
            if optimize_payload is not None:
                if not isinstance(optimize_payload, dict):
                    raise ValueError("optimization_candidate must be an object when provided")
                self._optimization_candidate = self._candidate_from_payload(
                    optimize_payload,
                    source="scripted_backend_optimization",
                )

    def _candidate_from_payload(self, payload: Dict[str, Any], *, source: str) -> Candidate:
        for required_key in ("candidate_id", "summary", "edits"):
            if required_key not in payload:
                raise ValueError(f"scripted candidate is missing required key: {required_key}")
        edits = []
        for item in payload["edits"]:
            edits.append(
                FileEdit(
                    path=str(item["path"]),
                    new_content=str(item["new_content"]),
                )
            )
        metadata = {"script_path": self._script_path}
        supported_classes = payload.get("supported_failure_classes")
        if supported_classes is not None:
            metadata["supported_failure_classes"] = tuple(str(item) for item in supported_classes)
        return Candidate(
            candidate_id=str(payload["candidate_id"]),
            summary=str(payload["summary"]),
            edits=tuple(edits),
            source=source,
            metadata=metadata,
        )

    def generate_initial_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
    ) -> Candidate:
        return self._initial_candidate

    def generate_repair_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
        critique: Critique,
        previous_candidate: Candidate,
        attempt_history: Sequence[AttemptRecord],
    ) -> Optional[Candidate]:
        while self._repair_cursor < len(self._repair_candidates):
            candidate = self._repair_candidates[self._repair_cursor]
            self._repair_cursor += 1
            supported = candidate.metadata.get("supported_failure_classes")
            if not supported or critique.failure_class in supported:
                return candidate
        return None

    def generate_optimization_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
        winning_candidate: Candidate,
        attempt_history: Sequence[AttemptRecord],
    ) -> Optional[Candidate]:
        return self._optimization_candidate


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
        "reason": "configured backend kind is unsupported in Phase 2",
    }
