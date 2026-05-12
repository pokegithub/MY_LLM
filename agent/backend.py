"""Backend protocol and fail-closed loaders for the verified coding agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from agent.types import (
    ROUTE_CODING,
    AttemptRecord,
    Candidate,
    ContextBundle,
    ContextFile,
    Critique,
    FileEdit,
    SolvePlan,
    TaskRequest,
)
from config import agent_cfg
from security.validator import ValidationError, get_allowed_data_roots, safe_load_json, validate_local_path


BACKEND_KIND_LOCAL_TRANSFORMERS = "local_transformers_in_process"
BACKEND_KIND_TRANSFORMER_ALIASES = {BACKEND_KIND_LOCAL_TRANSFORMERS}


class BackendError(RuntimeError):
    """Base error for truthful backend failure surfaces."""

    def __init__(self, failure_class: str, message: str):
        super().__init__(f"{failure_class}: {message}")
        self.failure_class = failure_class
        self.message = message


class BackendLoadError(BackendError):
    """Raised when a configured backend cannot be loaded."""


class BackendCandidateError(BackendError):
    """Raised when a backend cannot provide a usable candidate."""


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
        self._workspace_root = str(Path(workspace_root).resolve())
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
                metadata={"backend_kind": "scripted"},
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
                metadata={"backend_kind": "scripted"},
            )
            repair_payloads = self._payload.get("repair_candidates", [])
            if repair_payloads is None:
                repair_payloads = []
            if not isinstance(repair_payloads, list):
                raise ValueError("repair_candidates must be a list when provided")
            self._repair_candidates = tuple(
                self._candidate_from_payload(
                    item,
                    source="scripted_backend_repair",
                    metadata={"backend_kind": "scripted"},
                )
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
                    metadata={"backend_kind": "scripted"},
                )

    def _candidate_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        source: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Candidate:
        extra = {"script_path": self._script_path}
        if metadata:
            extra.update(dict(metadata))
        return candidate_from_payload(
            payload,
            source=source,
            workspace_root=self._workspace_root,
            metadata=extra,
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


def _validation_failure(message: str) -> BackendCandidateError:
    return BackendCandidateError("schema_validation_failed", message)


def _validate_candidate_edit_path(raw_path: str, workspace_root: str) -> str:
    path_text = str(raw_path).strip()
    if not path_text:
        raise _validation_failure("candidate edit path is required")
    candidate_path = Path(path_text)
    if candidate_path.is_absolute():
        raise _validation_failure("candidate edit path must be workspace-relative")
    if any(part == ".." for part in candidate_path.parts):
        raise _validation_failure("candidate edit path must not contain parent traversal")
    try:
        validate_local_path(
            Path(workspace_root) / candidate_path,
            allowed_roots=(workspace_root,),
            must_exist=False,
        )
    except ValidationError as exc:
        raise _validation_failure(str(exc)) from exc
    return path_text


def candidate_from_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
    workspace_root: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Candidate:
    """Validate a machine-readable payload and return a bounded candidate."""
    if not isinstance(payload, Mapping):
        raise _validation_failure("candidate payload must be a JSON object")
    for required_key in ("candidate_id", "summary", "edits"):
        if required_key not in payload:
            raise _validation_failure(f"candidate is missing required key: {required_key}")

    candidate_id = str(payload["candidate_id"]).strip()
    summary = str(payload["summary"]).strip()
    edits_payload = payload["edits"]
    if not candidate_id:
        raise _validation_failure("candidate_id must be non-empty")
    if not summary:
        raise _validation_failure("summary must be non-empty")
    if not isinstance(edits_payload, list) or not edits_payload:
        raise _validation_failure("edits must be a non-empty list")

    edits = []
    for item in edits_payload:
        if not isinstance(item, Mapping):
            raise _validation_failure("each edit must be a JSON object")
        if "path" not in item or "new_content" not in item:
            raise _validation_failure("each edit requires path and new_content")
        if not isinstance(item["new_content"], str):
            raise _validation_failure("edit new_content must be a string")
        edit_path = _validate_candidate_edit_path(str(item["path"]), workspace_root)
        edits.append(FileEdit(path=edit_path, new_content=item["new_content"]))

    candidate_metadata: Dict[str, Any] = dict(metadata or {})
    supported_classes = payload.get("supported_failure_classes")
    if supported_classes is not None:
        if not isinstance(supported_classes, list):
            raise _validation_failure("supported_failure_classes must be a list when provided")
        candidate_metadata["supported_failure_classes"] = tuple(str(item) for item in supported_classes)

    return Candidate(
        candidate_id=candidate_id,
        summary=summary,
        edits=tuple(edits),
        source=source,
        metadata=candidate_metadata,
    )


def candidate_from_model_text(
    text: str,
    *,
    source: str,
    workspace_root: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Candidate:
    """Parse strict JSON-only backend output into a Candidate."""
    raw = str(text).strip()
    if not raw:
        raise BackendCandidateError("malformed_candidate_output", "backend returned empty output")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendCandidateError(
            "malformed_candidate_output",
            f"backend output was not strict JSON: line={exc.lineno} col={exc.colno}",
        ) from exc
    return candidate_from_payload(
        payload,
        source=source,
        workspace_root=workspace_root,
        metadata=metadata,
    )


def _import_transformers_module():
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise BackendLoadError(
            "backend_unavailable",
            "transformers package is not installed in this environment",
        ) from exc
    return transformers


def _local_model_path_exists(model_id_or_path: Optional[str]) -> Optional[bool]:
    if not model_id_or_path:
        return None
    raw = str(model_id_or_path).strip()
    path = Path(raw).expanduser()
    if raw.startswith((".", "~")) or path.is_absolute():
        return path.exists()
    return None


def _compact_context(context: ContextBundle) -> str:
    chunks = [f"workspace_root: {context.workspace_root}"]
    if context.notes:
        chunks.append("notes:\n" + "\n".join(f"- {note}" for note in context.notes))
    for file_item in context.files:
        chunks.append(
            "\n".join(
                [
                    f"file: {file_item.path}",
                    f"exists: {file_item.exists}",
                    f"included: {file_item.included}",
                    "content_excerpt:",
                    file_item.content_excerpt,
                ]
            )
        )
    text = "\n\n".join(chunks)
    limit = max(512, int(agent_cfg.backend_prompt_max_chars))
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[context truncated by backend_prompt_max_chars]"


def _plan_summary(plan: SolvePlan) -> Dict[str, Any]:
    return {
        "route": plan.route,
        "target_files": list(plan.target_files),
        "checks": [
            {"check_type": check.check_type, "spec": check.spec, "meaningful": check.meaningful}
            for check in plan.checks
        ],
        "success_criteria": list(plan.success_criteria),
        "retry_budget": plan.retry_budget,
        "stop_conditions": list(plan.stop_conditions),
    }


def _attempt_history_summary(attempt_history: Sequence[AttemptRecord]) -> Sequence[Dict[str, Any]]:
    return tuple(
        {
            "attempt_index": attempt.attempt_index,
            "phase": attempt.phase,
            "origin": attempt.origin,
            "candidate_id": attempt.candidate_id,
            "kept": attempt.kept,
            "verification_passed": attempt.verification.overall_passed,
            "verification_summary": attempt.verification.summary,
            "critique_failure_class": attempt.critique.failure_class if attempt.critique else None,
            "critique_evidence": attempt.critique.evidence_summary if attempt.critique else None,
        }
        for attempt in attempt_history
    )


def _candidate_contract_prompt(
    *,
    purpose: str,
    request: TaskRequest,
    context: ContextBundle,
    plan: SolvePlan,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    payload = {
        "purpose": purpose,
        "task": request.task_text,
        "file_hints": list(request.file_hints),
        "plan": _plan_summary(plan),
        "context": _compact_context(context),
        "extra": dict(extra or {}),
        "candidate_json_schema": {
            "candidate_id": "short stable id",
            "summary": "one sentence describing the intended edit",
            "edits": [
                {
                    "path": "workspace-relative path",
                    "new_content": "full replacement file content",
                }
            ],
            "supported_failure_classes": ["optional", "repair", "classes"],
        },
        "constraints": [
            "Return strict JSON only. Do not wrap it in markdown.",
            "Use only workspace-relative file paths.",
            "Make the smallest relevant full-file replacement edit.",
            "Do not edit unrelated files.",
            "Do not claim correctness; the verifier decides success.",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _input_length(encoded: Mapping[str, Any]) -> int:
    input_ids = encoded.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape:
        return int(shape[-1])
    try:
        return len(input_ids[0])
    except (TypeError, IndexError, KeyError):
        return 0


def _move_encoded(encoded: Mapping[str, Any], device: str) -> Mapping[str, Any]:
    if device == "auto":
        return encoded
    moved: Dict[str, Any] = {}
    for key, value in encoded.items():
        to_fn = getattr(value, "to", None)
        moved[key] = to_fn(device) if callable(to_fn) else value
    return moved


class LocalTransformersInProcessBackend:
    """One explicit local Transformers backend path for real candidate generation."""

    def __init__(self, *, workspace_root: str):
        self.workspace_root = str(Path(workspace_root).resolve())
        self.model_id_or_path = str(agent_cfg.backend_model_id_or_path or "").strip()
        if not self.model_id_or_path:
            raise BackendLoadError(
                "backend_not_configured",
                "agent.backend_model_id_or_path is required for local Transformers backend",
            )
        transformers = _import_transformers_module()
        self.backend_kind = BACKEND_KIND_LOCAL_TRANSFORMERS
        self.device = str(agent_cfg.backend_device or "auto").strip().lower()
        self.max_new_tokens = int(agent_cfg.backend_max_new_tokens)
        self.temperature = float(agent_cfg.backend_temperature)
        self.local_files_only = bool(agent_cfg.backend_local_files_only)
        self.trust_remote_code = bool(agent_cfg.backend_trust_remote_code)
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id_or_path,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
            if self.device != "auto":
                to_fn = getattr(self._model, "to", None)
                if callable(to_fn):
                    self._model = to_fn(self.device)
            eval_fn = getattr(self._model, "eval", None)
            if callable(eval_fn):
                eval_fn()
        except BackendLoadError:
            raise
        except Exception as exc:
            raise BackendLoadError(
                "model_load_failed",
                f"local Transformers model could not be loaded: {exc}",
            ) from exc

    def _generate_text(self, prompt: str) -> str:
        try:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            if not isinstance(encoded, Mapping):
                encoded = dict(encoded)
            encoded = _move_encoded(encoded, self.device)
            start = _input_length(encoded)
            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature
            outputs = self._model.generate(**dict(encoded), **generate_kwargs)
            sequence = outputs[0]
            generated = sequence[start:] if start else sequence
            return str(self._tokenizer.decode(generated, skip_special_tokens=True)).strip()
        except BackendCandidateError:
            raise
        except Exception as exc:
            raise BackendCandidateError(
                "runtime_execution_failed",
                f"local Transformers generation failed: {exc}",
            ) from exc

    def _generate_candidate(
        self,
        *,
        source: str,
        prompt: str,
    ) -> Candidate:
        raw_text = self._generate_text(prompt)
        return candidate_from_model_text(
            raw_text,
            source=source,
            workspace_root=self.workspace_root,
            metadata={
                "backend_kind": self.backend_kind,
                "model_id_or_path": self.model_id_or_path,
                "local_files_only": self.local_files_only,
                "raw_output_chars": len(raw_text),
            },
        )

    def generate_initial_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
    ) -> Candidate:
        return self._generate_candidate(
            source="local_transformers_initial",
            prompt=_candidate_contract_prompt(
                purpose="initial_candidate",
                request=request,
                context=context,
                plan=plan,
            ),
        )

    def generate_repair_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
        critique: Critique,
        previous_candidate: Candidate,
        attempt_history: Sequence[AttemptRecord],
    ) -> Optional[Candidate]:
        return self._generate_candidate(
            source="local_transformers_repair",
            prompt=_candidate_contract_prompt(
                purpose="repair_candidate",
                request=request,
                context=context,
                plan=plan,
                extra={
                    "critique": {
                        "failure_class": critique.failure_class,
                        "root_cause": critique.root_cause,
                        "repair_targets": list(critique.repair_targets),
                        "evidence_summary": critique.evidence_summary,
                    },
                    "previous_candidate": {
                        "candidate_id": previous_candidate.candidate_id,
                        "summary": previous_candidate.summary,
                        "files": [edit.path for edit in previous_candidate.edits],
                    },
                    "attempt_history": _attempt_history_summary(attempt_history),
                },
            ),
        )

    def generate_optimization_candidate(
        self,
        request: TaskRequest,
        context: ContextBundle,
        plan: SolvePlan,
        winning_candidate: Candidate,
        attempt_history: Sequence[AttemptRecord],
    ) -> Optional[Candidate]:
        return self._generate_candidate(
            source="local_transformers_optimization",
            prompt=_candidate_contract_prompt(
                purpose="post_green_optimization_candidate",
                request=request,
                context=context,
                plan=plan,
                extra={
                    "winning_candidate": {
                        "candidate_id": winning_candidate.candidate_id,
                        "summary": winning_candidate.summary,
                        "files": [edit.path for edit in winning_candidate.edits],
                    },
                    "attempt_history": _attempt_history_summary(attempt_history),
                    "constraint": "Only propose a safe small improvement. The same verifier will rerun and regressions roll back.",
                },
            ),
        )


def backend_smoke_report(*, workspace_root: str = ".") -> Dict[str, Any]:
    """Run a narrow backend capability smoke check without applying edits."""
    backend, status = load_backend(workspace_root=workspace_root)
    report: Dict[str, Any] = {
        "schema": "agent_backend_smoke_v1",
        "backend_kind": status.get("kind", "unknown"),
        "configured_model": status.get("model_id_or_path") or getattr(agent_cfg, "backend_model_id_or_path", None),
        "backend_available": bool(status.get("available")),
        "load_status": "available" if status.get("available") else "failed",
        "local_files_only": status.get("local_files_only"),
        "model_path_exists": status.get("model_path_exists"),
        "structured_output_parse_status": "not_attempted",
        "tiny_candidate_generation_status": "not_attempted",
        "failure_class": status.get("failure_class"),
        "failure_reason": status.get("reason"),
        "proves_real_coding_ability": False,
        "quality_claim": "none",
    }
    if backend is None:
        return report

    request = TaskRequest(
        task_text="Backend smoke: produce a minimal candidate for backend_smoke_target.py",
        file_hints=("backend_smoke_target.py",),
        checks=(),
        workspace_root=workspace_root,
    )
    context = ContextBundle(
        workspace_root=str(Path(workspace_root).resolve()),
        files_considered=("backend_smoke_target.py",),
        files=(
            ContextFile(
                path="backend_smoke_target.py",
                exists=True,
                included=True,
                size_bytes=26,
                content_excerpt="def value():\n    return 1\n",
                note="smoke-only virtual file; no edit is applied",
            ),
        ),
        notes=("Backend smoke validates load and candidate schema only.",),
    )
    plan = SolvePlan(
        route=ROUTE_CODING,
        subgoals=("generate one bounded structured candidate",),
        target_files=("backend_smoke_target.py",),
        checks=(),
        risk_level="low",
        success_criteria=("candidate output parses into Candidate/FileEdit schema",),
        retry_budget=0,
        stop_conditions=("malformed output blocks the smoke check",),
    )
    try:
        candidate = backend.generate_initial_candidate(request, context, plan)
    except BackendCandidateError as exc:
        report["structured_output_parse_status"] = "failed"
        report["tiny_candidate_generation_status"] = "failed"
        report["failure_class"] = exc.failure_class
        report["failure_reason"] = exc.message
        return report
    except Exception as exc:
        report["structured_output_parse_status"] = "failed"
        report["tiny_candidate_generation_status"] = "failed"
        report["failure_class"] = "runtime_execution_failed"
        report["failure_reason"] = str(exc)
        return report

    report["structured_output_parse_status"] = "passed"
    report["tiny_candidate_generation_status"] = "candidate_generated"
    report["candidate_summary"] = {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "edit_count": len(candidate.edits),
        "edit_paths": [edit.path for edit in candidate.edits],
    }
    return report


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
                "failure_class": "backend_unavailable",
                "reason": "backend_script_path does not exist",
                "script_path": script_path,
            }
        try:
            backend = ScriptedCandidateBackend(script_path, workspace_root=workspace_root)
        except BackendCandidateError as exc:
            return None, {
                "configured": True,
                "available": False,
                "kind": "scripted",
                "failure_class": exc.failure_class,
                "reason": exc.message,
                "script_path": script_path,
            }
        except (OSError, ValueError, KeyError, ValidationError) as exc:
            return None, {
                "configured": True,
                "available": False,
                "kind": "scripted",
                "failure_class": "schema_validation_failed",
                "reason": str(exc),
                "script_path": script_path,
            }
        return (
            backend,
            {
                "configured": True,
                "available": True,
                "kind": "scripted",
                "script_path": str(Path(script_path).resolve()),
                "failure_class": None,
            },
        )

    if backend_kind == "none":
        return None, {
            "configured": False,
            "available": False,
            "kind": "none",
            "failure_class": "backend_not_configured",
            "reason": "no coding backend configured",
        }

    if backend_kind in BACKEND_KIND_TRANSFORMER_ALIASES:
        try:
            backend = LocalTransformersInProcessBackend(workspace_root=workspace_root)
        except BackendLoadError as exc:
            return None, {
                "configured": True,
                "available": False,
                "kind": BACKEND_KIND_LOCAL_TRANSFORMERS,
                "failure_class": exc.failure_class,
                "reason": exc.message,
                "model_id_or_path": agent_cfg.backend_model_id_or_path,
                "local_files_only": agent_cfg.backend_local_files_only,
                "model_path_exists": _local_model_path_exists(agent_cfg.backend_model_id_or_path),
            }
        return (
            backend,
            {
                "configured": True,
                "available": True,
                "kind": BACKEND_KIND_LOCAL_TRANSFORMERS,
                "failure_class": None,
                "model_id_or_path": backend.model_id_or_path,
                "local_files_only": backend.local_files_only,
                "model_path_exists": _local_model_path_exists(backend.model_id_or_path),
            },
        )

    return None, {
        "configured": True,
        "available": False,
        "kind": backend_kind,
        "failure_class": "unsupported_capability",
        "reason": "configured backend kind is unsupported by the current agent backend loader",
    }
