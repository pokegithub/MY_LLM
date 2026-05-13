"""Backend protocol and fail-closed loaders for the verified coding agent."""

from __future__ import annotations

import json
import os
import hashlib
import importlib.util
import sys
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
DEFAULT_TINY_SMOKE_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
DEFAULT_TINY_SMOKE_MODEL_DEST = "./run_artifacts/local_models/tiny-transformers-smoke"
DEFAULT_SMALL_CANDIDATE_CONFIG = "./configs/backend_smoke_small_candidate.json"
ALLOWED_TINY_SMOKE_MODEL_IDS = {
    "hf-internal-testing/tiny-random-gpt2",
    "sshleifer/tiny-gpt2",
}
SMALL_CANDIDATE_MODEL_OPTIONS = (
    {
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "destination": "./run_artifacts/local_models/smollm2-135m-instruct",
        "selection_reason": "first priority: very small open instruction-following smoke model",
        "max_download_target_mb": 750,
    },
    {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "destination": "./run_artifacts/local_models/qwen2.5-coder-0.5b-instruct",
        "selection_reason": "second priority: small coding-focused instruction smoke model",
        "max_download_target_mb": 1800,
    },
)
ALLOWED_SMALL_CANDIDATE_MODEL_IDS = {item["model_id"] for item in SMALL_CANDIDATE_MODEL_OPTIONS}
STRUCTURED_CANDIDATE_CONTRACT_VERSION = "structured_candidate_contract_v1"
MAX_FORMAT_RETRY_ATTEMPTS = 2


class BackendError(RuntimeError):
    """Base error for truthful backend failure surfaces."""

    def __init__(
        self,
        failure_class: str,
        message: str,
        *,
        failure_stage: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(f"{failure_class}: {message}")
        self.failure_class = failure_class
        self.message = message
        self.failure_stage = failure_stage
        self.details = dict(details or {})


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


def structured_candidate_schema() -> Dict[str, Any]:
    """Return the canonical JSON contract for backend-generated candidates."""
    return {
        "contract_version": STRUCTURED_CANDIDATE_CONTRACT_VERSION,
        "type": "object",
        "required": ["candidate_id", "summary", "edits"],
        "additional_properties": "ignored_except_supported_failure_classes",
        "properties": {
            "candidate_id": {
                "type": "string",
                "required": True,
                "description": "Non-empty short stable id for this candidate.",
            },
            "summary": {
                "type": "string",
                "required": True,
                "description": "Non-empty one sentence edit summary. This is not a correctness claim.",
            },
            "edits": {
                "type": "array",
                "required": True,
                "min_items": 1,
                "items": {
                    "type": "object",
                    "required": ["path", "new_content"],
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path only. Absolute paths and parent traversal are rejected.",
                        },
                        "new_content": {
                            "type": "string",
                            "description": "Full replacement file content. Empty strings are rejected in v1.",
                        },
                    },
                },
            },
            "supported_failure_classes": {
                "type": "array",
                "required": False,
                "description": "Optional repair-scoping labels only.",
            },
        },
        "rejected_cases": [
            "plain prose",
            "markdown fenced JSON",
            "partial JSON",
            "missing required fields",
            "absolute paths",
            "parent directory traversal",
            "empty edit list",
            "empty new_content",
        ],
    }


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
        if not item["new_content"].strip():
            raise _validation_failure("edit new_content must be non-empty")
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


def _format_retry_prompt(
    *,
    base_prompt: str,
    previous_output: str,
    parse_error: BackendCandidateError,
    retry_index: int,
) -> str:
    payload = {
        "purpose": "format_repair_only",
        "retry_index": retry_index,
        "parse_or_schema_error": {
            "failure_class": parse_error.failure_class,
            "message": parse_error.message,
        },
        "required_contract_version": STRUCTURED_CANDIDATE_CONTRACT_VERSION,
        "candidate_json_schema": structured_candidate_schema(),
        "original_candidate_request": base_prompt,
        "previous_invalid_output_excerpt": str(previous_output)[:1200],
        "constraints": [
            "Return exactly one JSON object and nothing else.",
            "Do not wrap the JSON in markdown.",
            "Do not explain the fix.",
            "Do not invent a new task or new target file.",
            "Use only the same task context and allowed file paths from the original request.",
            "Verifier decides correctness; do not claim success.",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _import_transformers_module():
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise BackendLoadError(
            "backend_unavailable",
            "transformers package is not installed in this environment",
            failure_stage="runtime_import",
        ) from exc
    return transformers


def transformers_runtime_status() -> Dict[str, Any]:
    """Return optional Transformers runtime availability without importing a model."""
    existing = sys.modules.get("transformers")
    if existing is not None:
        return {
            "available": True,
            "version": getattr(existing, "__version__", "unknown"),
            "failure_class": None,
            "reason": None,
        }
    try:
        spec = importlib.util.find_spec("transformers")
    except ValueError:
        spec = None
    if spec is None:
        return {
            "available": False,
            "failure_class": "backend_unavailable",
            "reason": "transformers package is not installed in this environment",
        }
    try:
        import transformers  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "failure_class": "backend_unavailable",
            "reason": f"transformers package could not be imported: {exc}",
        }
    return {
        "available": True,
        "version": getattr(transformers, "__version__", "unknown"),
        "failure_class": None,
        "reason": None,
    }


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
        "required_contract_version": STRUCTURED_CANDIDATE_CONTRACT_VERSION,
        "candidate_json_schema": structured_candidate_schema(),
        "allowed_target_files": list(plan.target_files or request.file_hints),
        "constraints": [
            "Return strict JSON only. No markdown fences, no prose, no comments.",
            "Use only workspace-relative file paths.",
            "Use only allowed_target_files unless the plan explicitly requires another workspace file.",
            "Make the smallest relevant full-file replacement edit.",
            "Do not edit unrelated files.",
            "Do not claim correctness; the verifier decides success.",
            "Do not include hidden reasoning.",
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
        except Exception as exc:
            raise BackendLoadError(
                "model_load_failed",
                f"tokenizer load failed: {exc}",
                failure_stage="tokenizer_load",
            ) from exc
        try:
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
                f"model load failed: {exc}",
                failure_stage="model_load",
            ) from exc

    def _max_input_tokens(self) -> Optional[int]:
        values = []
        config = getattr(self._model, "config", None)
        for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                values.append(value)
        tokenizer_limit = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            values.append(tokenizer_limit)
        if not values:
            return None
        context_limit = min(values)
        return max(1, context_limit - max(1, self.max_new_tokens))

    def _generate_text(self, prompt: str) -> str:
        try:
            max_input_tokens = self._max_input_tokens()
            tokenizer_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
            if max_input_tokens is not None:
                tokenizer_kwargs.update({"truncation": True, "max_length": max_input_tokens})
            encoded = self._tokenizer(prompt, **tokenizer_kwargs)
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
        raw_attempts = []
        current_prompt = prompt
        last_error: Optional[BackendCandidateError] = None
        for attempt_index in range(MAX_FORMAT_RETRY_ATTEMPTS + 1):
            raw_text = self._generate_text(current_prompt)
            raw_attempts.append(
                {
                    "attempt_index": attempt_index,
                    "raw_output_chars": len(raw_text),
                    "raw_output_sha256": hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest(),
                }
            )
            try:
                candidate = candidate_from_model_text(
                    raw_text,
                    source=source,
                    workspace_root=self.workspace_root,
                    metadata={
                        "backend_kind": self.backend_kind,
                        "model_id_or_path": self.model_id_or_path,
                        "local_files_only": self.local_files_only,
                        "raw_output_chars": len(raw_text),
                        "structured_contract_version": STRUCTURED_CANDIDATE_CONTRACT_VERSION,
                        "generation_attempts": attempt_index + 1,
                        "malformed_retry_count": attempt_index,
                        "structured_output_parse_status": "passed",
                        "structured_candidate_valid": True,
                        "format_retry_attempts": tuple(raw_attempts),
                    },
                )
                return candidate
            except BackendCandidateError as exc:
                last_error = exc
                raw_attempts[-1]["failure_class"] = exc.failure_class
                raw_attempts[-1]["failure_reason"] = exc.message
                if attempt_index >= MAX_FORMAT_RETRY_ATTEMPTS:
                    break
                current_prompt = _format_retry_prompt(
                    base_prompt=prompt,
                    previous_output=raw_text,
                    parse_error=exc,
                    retry_index=attempt_index + 1,
                )

        assert last_error is not None
        raise BackendCandidateError(
            last_error.failure_class,
            last_error.message,
            details={
                "structured_contract_version": STRUCTURED_CANDIDATE_CONTRACT_VERSION,
                "generation_attempts": len(raw_attempts),
                "malformed_retry_count": max(0, len(raw_attempts) - 1),
                "structured_output_parse_status": "failed",
                "structured_candidate_valid": False,
                "format_retry_attempts": tuple(raw_attempts),
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


def _smoke_request_context_plan(workspace_root: str) -> tuple[TaskRequest, ContextBundle, SolvePlan]:
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
    return request, context, plan


def _smoke_load_status(report: Dict[str, Any], status: Mapping[str, Any]) -> None:
    stage = status.get("failure_stage")
    if status.get("available"):
        report["tokenizer_load_status"] = "passed"
        report["model_load_status"] = "passed"
        report["smoke_level"] = "model_loaded"
        return
    if not report.get("transformers_available"):
        report["smoke_level"] = "runtime_unavailable"
        return
    if status.get("model_path_exists") is False and status.get("local_files_only") is True:
        report["smoke_level"] = "model_missing"
    elif stage == "tokenizer_load":
        report["smoke_level"] = "model_load_failed"
    elif stage == "model_load":
        report["tokenizer_load_status"] = "passed"
        report["smoke_level"] = "model_load_failed"
    else:
        report["smoke_level"] = "runtime_unavailable"
    if stage == "tokenizer_load":
        report["tokenizer_load_status"] = "failed"
    if stage == "model_load":
        report["model_load_status"] = "failed"


def _unsafe_path_detected(failure_class: Optional[str], message: Optional[str]) -> bool:
    if failure_class != "schema_validation_failed":
        return False
    text = str(message or "").lower()
    return any(token in text for token in ("path", "absolute", "parent traversal", "workspace-relative"))


def backend_smoke_report(*, workspace_root: str = ".") -> Dict[str, Any]:
    """Run a narrow backend capability smoke check without applying edits."""
    runtime = transformers_runtime_status()
    backend, status = load_backend(workspace_root=workspace_root)
    report: Dict[str, Any] = {
        "schema": "agent_backend_smoke_v1",
        "backend_kind": status.get("kind", "unknown"),
        "configured_model": status.get("model_id_or_path") or getattr(agent_cfg, "backend_model_id_or_path", None),
        "backend_available": bool(status.get("available")),
        "load_status": "available" if status.get("available") else "failed",
        "local_files_only": status.get("local_files_only"),
        "model_path_exists": status.get("model_path_exists"),
        "transformers_available": bool(runtime.get("available")),
        "transformers_version": runtime.get("version"),
        "tokenizer_load_status": "not_attempted",
        "model_load_status": "not_attempted",
        "generation_status": "not_attempted",
        "generated_text_chars": 0,
        "generated_text_sample": None,
        "generated_text_sha256": None,
        "structured_contract_version": STRUCTURED_CANDIDATE_CONTRACT_VERSION,
        "generation_attempts": 0,
        "malformed_retry_count": 0,
        "final_parse_status": "not_attempted",
        "final_schema_validation_status": "not_attempted",
        "structured_output_parse_status": "not_attempted",
        "structured_candidate_valid": False,
        "candidate_valid": False,
        "rejected_reason": None,
        "unsafe_path_detected": False,
        "tiny_candidate_generation_status": "not_attempted",
        "failure_class": status.get("failure_class"),
        "failure_reason": status.get("reason"),
        "smoke_level": "not_started",
        "proves_real_coding_ability": False,
        "quality_claim": "none",
    }
    _smoke_load_status(report, status)
    if backend is None:
        return report

    request, context, plan = _smoke_request_context_plan(workspace_root)
    try:
        if isinstance(backend, LocalTransformersInProcessBackend):
            prompt = _candidate_contract_prompt(
                purpose="initial_candidate",
                request=request,
                context=context,
                plan=plan,
            )
            candidate = backend._generate_candidate(
                source="local_transformers_initial",
                prompt=prompt,
            )
            report["generation_status"] = "passed"
        else:
            candidate = backend.generate_initial_candidate(request, context, plan)
            report["generation_status"] = "passed"
    except BackendCandidateError as exc:
        details = dict(exc.details or {})
        report["generation_attempts"] = int(details.get("generation_attempts") or (1 if exc.failure_class != "runtime_execution_failed" else 0))
        report["malformed_retry_count"] = int(details.get("malformed_retry_count") or 0)
        attempts = details.get("format_retry_attempts") or ()
        if attempts:
            last_attempt = attempts[-1]
            if isinstance(last_attempt, Mapping):
                report["generated_text_chars"] = int(last_attempt.get("raw_output_chars") or 0)
                report["generated_text_sha256"] = last_attempt.get("raw_output_sha256")
        if report["generation_attempts"] > 0 and exc.failure_class != "runtime_execution_failed":
            report["generation_status"] = "passed"
        report["structured_output_parse_status"] = "failed"
        report["final_parse_status"] = "failed"
        report["final_schema_validation_status"] = "failed" if exc.failure_class == "schema_validation_failed" else "not_attempted"
        report["tiny_candidate_generation_status"] = "failed"
        report["failure_class"] = exc.failure_class
        report["failure_reason"] = exc.message
        report["rejected_reason"] = exc.message
        report["unsafe_path_detected"] = _unsafe_path_detected(exc.failure_class, exc.message)
        if report["generation_status"] == "passed":
            if report["malformed_retry_count"] > 0:
                report["smoke_level"] = "model_loaded_generation_succeeded_parse_retried_failed"
            else:
                report["smoke_level"] = "model_loaded_generation_succeeded_parse_failed"
        else:
            report["generation_status"] = "failed"
            report["smoke_level"] = "model_loaded_generation_failed"
        return report
    except Exception as exc:
        report["structured_output_parse_status"] = "failed"
        report["tiny_candidate_generation_status"] = "failed"
        report["failure_class"] = "runtime_execution_failed"
        report["failure_reason"] = str(exc)
        report["rejected_reason"] = str(exc)
        report["generation_status"] = "failed"
        report["smoke_level"] = "model_loaded_generation_failed"
        return report

    report["structured_output_parse_status"] = "passed"
    report["final_parse_status"] = "passed"
    report["final_schema_validation_status"] = "passed"
    report["structured_candidate_valid"] = True
    report["candidate_valid"] = True
    report["generation_attempts"] = int(candidate.metadata.get("generation_attempts") or 1)
    report["malformed_retry_count"] = int(candidate.metadata.get("malformed_retry_count") or 0)
    report["tiny_candidate_generation_status"] = "candidate_generated"
    report["smoke_level"] = "model_loaded_generation_succeeded_parse_passed"
    report["candidate_summary"] = {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "edit_count": len(candidate.edits),
        "edit_paths": [edit.path for edit in candidate.edits],
    }
    return report


def _directory_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"file_count": 0, "size_bytes": 0}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
    }


def _looks_like_transformers_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_model = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    has_tokenizer = (path / "tokenizer.json").is_file() or (path / "tokenizer_config.json").is_file()
    return has_config and has_model and has_tokenizer


def _write_backend_config(path: str, model_path: Path, *, max_new_tokens: int = 512) -> str:
    config_path = Path(path).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    rel_model_path = os.path.relpath(model_path, Path(os.getcwd()).resolve()).replace("\\", "/")
    payload = {
        "agent": {
            "backend_kind": BACKEND_KIND_LOCAL_TRANSFORMERS,
            "backend_model_id_or_path": f"./{rel_model_path}",
            "backend_local_files_only": True,
            "backend_trust_remote_code": False,
            "backend_device": "auto",
            "backend_max_new_tokens": max_new_tokens,
            "backend_temperature": 0.0,
            "backend_prompt_max_chars": 8000,
        }
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(config_path)


def _download_and_save_transformers_model(
    *,
    transformers: Any,
    model_id: str,
    destination_path: Path,
    source_schema: str,
    source_payload: Mapping[str, Any],
) -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=False,
        trust_remote_code=False,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=False,
        trust_remote_code=False,
    )
    destination_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(destination_path))
    model.save_pretrained(str(destination_path), safe_serialization=True)
    payload = dict(source_payload)
    payload["schema"] = source_schema
    payload["model_id"] = model_id
    with open(destination_path / "backend_model_source.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _candidate_options_for(model_id: Optional[str], destination: Optional[str]) -> tuple[Dict[str, Any], ...]:
    if model_id:
        if model_id not in ALLOWED_SMALL_CANDIDATE_MODEL_IDS:
            return (
                {
                    "model_id": str(model_id),
                    "destination": destination or "./run_artifacts/local_models/unlisted-small-candidate",
                    "selection_reason": "rejected: model id is not in the small candidate allowlist",
                    "rejected": True,
                },
            )
        selected = next(item for item in SMALL_CANDIDATE_MODEL_OPTIONS if item["model_id"] == model_id)
        candidate = dict(selected)
        if destination:
            candidate["destination"] = destination
        return (candidate,)
    candidates = [dict(item) for item in SMALL_CANDIDATE_MODEL_OPTIONS]
    if destination and candidates:
        candidates[0]["destination"] = destination
    return tuple(candidates)


def provision_small_candidate_model(
    *,
    model_id: Optional[str] = None,
    destination: Optional[str] = None,
    config_path: str = DEFAULT_SMALL_CANDIDATE_CONFIG,
) -> Dict[str, Any]:
    """Provision the first safe small instruction-following backend smoke candidate."""
    report: Dict[str, Any] = {
        "schema": "agent_backend_small_candidate_provision_v1",
        "selection_policy": "priority_allowlist_small_instruction_models_only",
        "priority_model_ids": [item["model_id"] for item in SMALL_CANDIDATE_MODEL_OPTIONS],
        "selected_model_id": None,
        "destination": None,
        "config_path": str(Path(config_path).resolve()),
        "downloaded_autonomously": False,
        "internet_required": True,
        "local_files_only_after_provisioning": True,
        "smoke_only": True,
        "proves_model_quality": False,
        "bakeoff_winner_claim": "none",
        "quality_claim": "none",
        "status": "not_started",
        "ok": False,
        "attempts": [],
    }

    options = _candidate_options_for(model_id, destination)
    if options and options[0].get("rejected"):
        report.update(
            {
                "selected_model_id": options[0]["model_id"],
                "destination": str(Path(options[0]["destination"]).resolve()),
                "status": "rejected",
                "failure_class": "unsupported_capability",
                "failure_reason": "requested model id is not in the small instruction candidate allowlist",
                "manual_action_required": False,
            }
        )
        return report

    runtime = transformers_runtime_status()
    report["transformers_available"] = bool(runtime.get("available"))
    report["transformers_version"] = runtime.get("version")
    if not runtime.get("available"):
        report.update(
            {
                "status": "failed",
                "failure_class": runtime.get("failure_class"),
                "failure_reason": runtime.get("reason"),
                "manual_action_required": True,
            }
        )
        return report

    try:
        transformers = _import_transformers_module()
    except BackendLoadError as exc:
        report.update(
            {
                "status": "failed",
                "failure_class": exc.failure_class,
                "failure_reason": exc.message,
                "manual_action_required": True,
            }
        )
        return report

    workspace_root = Path(os.getcwd()).resolve()
    for option in options:
        destination_path = Path(str(option["destination"])).expanduser().resolve()
        attempt: Dict[str, Any] = {
            "model_id": option["model_id"],
            "destination": str(destination_path),
            "selection_reason": option.get("selection_reason"),
            "status": "not_started",
        }
        report["attempts"].append(attempt)
        try:
            validate_local_path(
                destination_path,
                allowed_roots=(workspace_root,),
                must_exist=False,
            )
        except ValidationError as exc:
            attempt.update(
                {
                    "status": "rejected",
                    "failure_class": "schema_validation_failed",
                    "failure_reason": str(exc),
                }
            )
            continue

        if _looks_like_transformers_model_dir(destination_path):
            config_written = _write_backend_config(config_path, destination_path)
            summary = _directory_summary(destination_path)
            attempt.update({"status": "already_present", "artifact_summary": summary})
            report.update(
                {
                    "selected_model_id": option["model_id"],
                    "destination": str(destination_path),
                    "config_path": config_written,
                    "downloaded_autonomously": False,
                    "status": "already_present",
                    "ok": True,
                    "artifact_summary": summary,
                    "manual_action_required": False,
                }
            )
            return report

        try:
            _download_and_save_transformers_model(
                transformers=transformers,
                model_id=option["model_id"],
                destination_path=destination_path,
                source_schema="backend_small_candidate_model_source_v1",
                source_payload={
                    "smoke_only": True,
                    "instruction_following_candidate": True,
                    "proves_model_quality": False,
                    "bakeoff_winner_claim": "none",
                    "quality_claim": "none",
                    "selection_reason": option.get("selection_reason"),
                    "max_download_target_mb": option.get("max_download_target_mb"),
                },
            )
            config_written = _write_backend_config(config_path, destination_path)
            summary = _directory_summary(destination_path)
            attempt.update({"status": "downloaded", "artifact_summary": summary})
            report.update(
                {
                    "selected_model_id": option["model_id"],
                    "destination": str(destination_path),
                    "config_path": config_written,
                    "downloaded_autonomously": True,
                    "status": "provisioned",
                    "ok": True,
                    "artifact_summary": summary,
                    "manual_action_required": False,
                }
            )
            return report
        except Exception as exc:
            attempt.update(
                {
                    "status": "failed",
                    "failure_class": "model_download_or_load_failed",
                    "failure_reason": str(exc),
                }
            )

    report.update(
        {
            "status": "failed",
            "failure_class": "all_small_candidate_options_failed",
            "failure_reason": "no allowlisted small instruction candidate could be provisioned autonomously",
            "manual_action_required": True,
            "manual_action": "Place a small ungated Transformers-compatible instruction model under run_artifacts/local_models/ and rerun provisioning.",
        }
    )
    return report


def provision_tiny_transformers_model(
    *,
    model_id: str = DEFAULT_TINY_SMOKE_MODEL_ID,
    destination: str = DEFAULT_TINY_SMOKE_MODEL_DEST,
    allow_non_tiny_model_id: bool = False,
) -> Dict[str, Any]:
    """Explicitly download and save a smoke-only tiny Transformers model."""
    model_id = str(model_id).strip()
    destination_path = Path(destination).expanduser().resolve()
    report: Dict[str, Any] = {
        "schema": "agent_backend_tiny_model_provision_v1",
        "model_id": model_id,
        "destination": str(destination_path),
        "allowed_tiny_model_ids": sorted(ALLOWED_TINY_SMOKE_MODEL_IDS),
        "internet_required": True,
        "smoke_only": True,
        "proves_real_coding_ability": False,
        "quality_claim": "none",
        "status": "not_started",
        "ok": False,
    }
    if model_id not in ALLOWED_TINY_SMOKE_MODEL_IDS and not allow_non_tiny_model_id:
        report.update(
            {
                "status": "rejected",
                "failure_class": "unsupported_capability",
                "failure_reason": "model id is not in the tiny smoke allowlist",
            }
        )
        return report
    try:
        validate_local_path(
            destination_path,
            allowed_roots=(Path(os.getcwd()).resolve(),),
            must_exist=False,
        )
    except ValidationError as exc:
        report.update(
            {
                "status": "rejected",
                "failure_class": "schema_validation_failed",
                "failure_reason": str(exc),
            }
        )
        return report

    runtime = transformers_runtime_status()
    report["transformers_available"] = bool(runtime.get("available"))
    report["transformers_version"] = runtime.get("version")
    if not runtime.get("available"):
        report.update(
            {
                "status": "failed",
                "failure_class": runtime.get("failure_class"),
                "failure_reason": runtime.get("reason"),
            }
        )
        return report

    try:
        transformers = _import_transformers_module()
        _download_and_save_transformers_model(
            transformers=transformers,
            model_id=model_id,
            destination_path=destination_path,
            source_schema="backend_smoke_model_source_v1",
            source_payload={
                "smoke_only": True,
                "proves_real_coding_ability": False,
                "quality_claim": "none",
            },
        )
        with open(destination_path / "backend_smoke_model_source.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": "backend_smoke_model_source_v1",
                    "model_id": model_id,
                    "smoke_only": True,
                    "proves_real_coding_ability": False,
                    "quality_claim": "none",
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_class": "model_load_failed",
                "failure_reason": str(exc),
            }
        )
        return report

    report.update(
        {
            "status": "provisioned",
            "ok": True,
            "local_files_only_for_smoke": True,
            "model_path_exists": destination_path.exists(),
            "artifact_summary": _directory_summary(destination_path),
        }
    )
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
                "failure_stage": exc.failure_stage,
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
