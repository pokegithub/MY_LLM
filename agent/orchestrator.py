"""Top-level orchestration for the verified coding agent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from agent.backend import BackendCandidateError, CodingModelBackend, load_backend
from agent.critic import critique_failure
from agent.context import build_context
from agent.exact_tools import execute_exact_task
from agent.optimize import run_optimizer
from agent.planner import build_plan
from agent.retrieval import summarize_for_context
from agent.router import route_task
from agent.trajectory import write_trajectory_record
from agent.types import (
    OPTIMIZATION_NOT_ATTEMPTED,
    PLAN_STATUS_READY,
    ROUTE_CODING,
    ROUTE_EXACT,
    AttemptRecord,
    Candidate,
    Critique,
    SOLVE_STATUS_BLOCKED,
    SOLVE_STATUS_FAILED,
    SOLVE_STATUS_UNSUPPORTED,
    SOLVE_STATUS_VERIFIED,
    ContextBundle,
    RouteDecision,
    SolveResult,
    TaskRequest,
    VerificationReport,
    to_dict,
)
from agent.verify import run_verification
from agent.workspace import WorkspaceEditSession
from config import agent_cfg
from observability.run_registry import RunRegistry


def _new_run_dir(operation: str) -> tuple[str, str]:
    run_id = f"{operation}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_dir = os.path.join(agent_cfg.report_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


def _write_report(run_dir: str, payload: Dict) -> str:
    path = os.path.join(run_dir, "report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _empty_verification(summary: str) -> VerificationReport:
    return VerificationReport(
        overall_passed=False,
        meaningful=False,
        checks=(),
        summary=summary,
        quality_claim="none",
    )


def _finalize(
    *,
    run_id: str,
    run_dir: str,
    operation: str,
    request: TaskRequest,
    route: RouteDecision,
    backend_status: Dict,
    plan,
    context: ContextBundle,
    verification: VerificationReport,
    status: str,
    files_touched: List[str],
    blocked_reason: Optional[str],
    candidate=None,
    retry_budget: int = 0,
    attempts: Sequence[AttemptRecord] = (),
    final_origin: str = "none",
    winning_attempt: Optional[int] = None,
    optimization_status: str = OPTIMIZATION_NOT_ATTEMPTED,
) -> Dict:
    result = SolveResult(
        run_id=run_id,
        operation=operation,
        request=request,
        route=route,
        backend_status=backend_status,
        plan=plan,
        context=context,
        verification=verification,
        status=status,
        files_touched=tuple(files_touched),
        blocked_reason=blocked_reason,
        quality_claim=verification.quality_claim if status == SOLVE_STATUS_VERIFIED else "none",
        report_path="",
        candidate=candidate,
        retry_budget=retry_budget,
        attempts=tuple(attempts),
        final_origin=final_origin,
        winning_attempt=winning_attempt,
        optimization_status=optimization_status,
    )
    payload = to_dict(result)
    payload["schema"] = "agent_phase2_report_v1"
    payload["timestamp"] = int(time.time())
    if attempts or status != SOLVE_STATUS_VERIFIED:
        payload["failure_diagnostics"] = _build_failure_diagnostics(
            status=status,
            request=request,
            run_dir=run_dir,
            backend_status=backend_status,
            retry_budget=retry_budget,
            attempts=tuple(attempts),
            winning_attempt=winning_attempt,
        )
    report_path = _write_report(run_dir, payload)
    payload["report_path"] = report_path
    registry_status = {
        "ok": True,
        "required": False,
        "non_critical": True,
    }
    artifact_warnings = []
    try:
        RunRegistry().add(
            f"agent_{operation}",
            {
                "run_id": run_id,
                "status": status,
                "route": route.route,
                "report_path": os.path.abspath(report_path),
            },
        )
    except (OSError, PermissionError, TimeoutError, ValueError) as exc:
        registry_status = {
            "ok": False,
            "error": str(exc),
            "required": False,
            "non_critical": True,
            "warning": "non-critical run registry write failed; primary agent report was still written",
        }
        artifact_warnings.append(
            {
                "target": "run_registry",
                "severity": "warning",
                "required": False,
                "message": str(exc),
            }
        )
    payload["run_registry"] = registry_status
    payload["degraded_mode"] = bool(artifact_warnings)
    payload["artifact_warnings"] = artifact_warnings
    trajectory_status = {
        "ok": True,
        "required": False,
        "non_critical": True,
    }
    try:
        trajectory_status = write_trajectory_record(payload)
    except (OSError, PermissionError, TimeoutError, ValueError) as exc:
        trajectory_status = {
            "ok": False,
            "error": str(exc),
            "required": False,
            "non_critical": True,
            "warning": "non-critical trajectory write failed; primary agent report was still written",
        }
        artifact_warnings.append(
            {
                "target": "trajectory_store",
                "severity": "warning",
                "required": False,
                "message": str(exc),
            }
        )
    payload["trajectory_store"] = trajectory_status
    payload["degraded_mode"] = bool(artifact_warnings)
    payload["artifact_warnings"] = artifact_warnings
    _write_report(run_dir, payload)
    return payload


def _empty_attempt_history() -> tuple[AttemptRecord, ...]:
    return ()


def _first_failed_check(verification: VerificationReport):
    for check in verification.checks:
        if not check.passed:
            return check
    return verification.checks[0] if verification.checks else None


def _short_text(text: str, *, limit: int = 320) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) > limit:
        return compact[: limit - 3].rstrip() + "..."
    return compact


def _stable_failure_signature(*, failure_class: str, check_type: str, spec: str, summary: str) -> str:
    normalized = "|".join(
        [
            str(failure_class or "unknown"),
            str(check_type or "unknown"),
            str(spec or ""),
            _short_text(summary, limit=240).lower(),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _attempt_failure_diagnostic(attempt: AttemptRecord) -> Dict[str, Any]:
    verification = attempt.verification
    failed_check = _first_failed_check(verification)
    failure_class = (
        attempt.critique.failure_class
        if attempt.critique is not None
        else ("none" if verification.overall_passed else "unverified_failure")
    )
    check_type = failed_check.check_type if failed_check else "none"
    check_spec = failed_check.spec if failed_check else ""
    check_summary = failed_check.summary if failed_check else verification.summary
    return {
        "attempt_index": attempt.attempt_index,
        "candidate_id": attempt.candidate_id,
        "origin": attempt.origin,
        "phase": attempt.phase,
        "files_touched": list(attempt.files_touched),
        "kept": bool(attempt.kept),
        "verifier_passed": bool(verification.overall_passed),
        "failure_class": failure_class,
        "verifier_command": failed_check.evidence.get("command") if failed_check and failed_check.evidence else None,
        "verifier_stdout_path": failed_check.stdout_path if failed_check else None,
        "verifier_stderr_path": failed_check.stderr_path if failed_check else None,
        "short_assertion_summary": _short_text(check_summary),
        "stable_failure_signature": _stable_failure_signature(
            failure_class=failure_class,
            check_type=check_type,
            spec=check_spec,
            summary=check_summary,
        ),
    }


def _failure_signature_repeated(failed_attempts: Sequence[Dict[str, Any]]) -> bool:
    signatures = [str(item.get("stable_failure_signature") or "") for item in failed_attempts]
    signatures = [item for item in signatures if item]
    return bool(signatures) and len(signatures) != len(set(signatures))


def _unique_text(items: Sequence[str]) -> List[str]:
    result: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def _candidate_edit_paths(candidate: Candidate) -> List[str]:
    return _unique_text([edit.path for edit in candidate.edits])


def _repair_feedback_context(
    *,
    critique: Critique,
    previous_candidate: Candidate,
    attempt_history: Sequence[AttemptRecord],
    plan_retry_budget: int,
) -> Dict[str, Any]:
    last_attempt = attempt_history[-1] if attempt_history else None
    failed_check = _first_failed_check(last_attempt.verification) if last_attempt is not None else None
    failed_summary = (
        failed_check.summary
        if failed_check is not None
        else (last_attempt.verification.summary if last_attempt is not None else critique.evidence_summary)
    )
    touched_files = _unique_text(
        list(last_attempt.files_touched if last_attempt is not None else ())
        + _candidate_edit_paths(previous_candidate)
        + list(critique.repair_targets)
    )
    failed_diagnostics = [
        _attempt_failure_diagnostic(attempt)
        for attempt in attempt_history
        if not attempt.verification.overall_passed
    ]
    repeated_signature = _failure_signature_repeated(failed_diagnostics)
    return {
        "schema": "agent_repair_prompt_feedback_v1",
        "failure_class": critique.failure_class,
        "failing_check_summary": _short_text(failed_summary),
        "failure_root_cause": _short_text(critique.root_cause),
        "touched_file_paths": touched_files,
        "previous_candidate": {
            "candidate_id": previous_candidate.candidate_id,
            "summary": _short_text(previous_candidate.summary),
            "edit_paths": _candidate_edit_paths(previous_candidate),
        },
        "attempts_already_failed": len(failed_diagnostics),
        "retry_budget": max(0, int(plan_retry_budget)),
        "repeated_failure_signature_detected": repeated_signature,
        "repair_instructions": [
            "Use the verifier feedback above as the repair target.",
            "Do not repeat failed behavior from the previous candidate.",
            "The verifier remains final authority; do not claim success in the candidate summary.",
            "Respect the bounded retry budget and make the smallest relevant full-file replacement.",
            "Preserve structured_candidate_contract_v1: return exactly one JSON object with candidate_id, summary, and edits.",
        ],
        "repeated_failed_behavior_warning": (
            "A previous verifier failure signature repeated; change the actual failing behavior instead of restating the same edit."
            if repeated_signature
            else None
        ),
    }


def _critique_with_repair_feedback(
    *,
    critique: Critique,
    previous_candidate: Candidate,
    attempt_history: Sequence[AttemptRecord],
    plan_retry_budget: int,
) -> Critique:
    feedback = _repair_feedback_context(
        critique=critique,
        previous_candidate=previous_candidate,
        attempt_history=attempt_history,
        plan_retry_budget=plan_retry_budget,
    )
    feedback_text = json.dumps(feedback, indent=2, sort_keys=True)
    repair_targets = tuple(_unique_text(list(critique.repair_targets) + feedback["touched_file_paths"]))
    return Critique(
        failure_class=critique.failure_class,
        root_cause=(
            f"{critique.root_cause}; repair prompt includes verifier feedback, "
            f"failure class, previous candidate summary, and retry-budget reminders"
        ),
        repair_targets=repair_targets,
        blocked_reason=critique.blocked_reason,
        confidence=critique.confidence,
        evidence_summary=f"{critique.evidence_summary}\n\nREPAIR_LOOP_FEEDBACK_V1:\n{feedback_text}",
    )


def _diagnostic_failure_class(attempts: Sequence[AttemptRecord], backend_status: Dict, status: str) -> str:
    failed_attempts = [attempt for attempt in attempts if not attempt.verification.overall_passed]
    if failed_attempts:
        last = failed_attempts[-1]
        if last.critique is not None:
            return str(last.critique.failure_class)
        return "verifier_failure"
    if backend_status.get("last_failure_class"):
        return str(backend_status.get("last_failure_class"))
    if backend_status.get("failure_class"):
        return str(backend_status.get("failure_class"))
    if status == SOLVE_STATUS_BLOCKED:
        return "blocked_before_verifier"
    if status == SOLVE_STATUS_FAILED:
        return "verification_failed"
    return "none"


def _verifier_failure_type(final_failure_class: str) -> str:
    mapping = {
        "assertion_failure": "verifier_assertion_failure",
        "behavior_mismatch": "verifier_behavior_failure",
        "syntax_error": "verifier_syntax_failure",
        "import_error": "verifier_import_failure",
        "runtime_error": "verifier_runtime_failure",
        "timeout": "verifier_timeout",
        "performance_regression": "verifier_performance_failure",
        "verifier_missing": "verifier_missing",
        "unsupported_assumption": "unsupported_verifier_contract",
    }
    if final_failure_class in mapping:
        return mapping[final_failure_class]
    if final_failure_class in {"backend_unavailable", "model_load_failed", "malformed_candidate_output", "schema_validation_failed"}:
        return "backend_or_candidate_generation_failure"
    if final_failure_class == "none":
        return "none"
    return "verifier_failure"


def _infrastructure_failure_detected(final_failure_class: str, backend_status: Dict) -> bool:
    infrastructure_classes = {
        "backend_unavailable",
        "model_load_failed",
        "malformed_candidate_output",
        "schema_validation_failed",
        "verifier_missing",
        "unsupported_assumption",
    }
    return final_failure_class in infrastructure_classes or bool(backend_status.get("failure_class"))


def _build_failure_diagnostics(
    *,
    status: str,
    request: TaskRequest,
    run_dir: str,
    backend_status: Dict,
    retry_budget: int,
    attempts: Sequence[AttemptRecord],
    winning_attempt: Optional[int],
) -> Dict[str, Any]:
    attempt_diagnostics = [_attempt_failure_diagnostic(attempt) for attempt in attempts]
    failed_attempts = [item for item in attempt_diagnostics if not item["verifier_passed"]]
    final_failure_class = _diagnostic_failure_class(attempts, backend_status, status)
    check_names = [
        str((attempt.verification.checks[0].check_type if attempt.verification.checks else "none"))
        for attempt in attempts
        if not attempt.verification.overall_passed
    ]
    evidence_paths: List[str] = []
    for item in attempt_diagnostics:
        for key in ("verifier_stdout_path", "verifier_stderr_path"):
            value = item.get(key)
            if value:
                evidence_paths.append(str(value))
    distinct_paths = bool(evidence_paths) and len(evidence_paths) == len(set(evidence_paths))
    last_failed = failed_attempts[-1] if failed_attempts else None
    repeated_failure_signature = _failure_signature_repeated(failed_attempts)
    repair_prompt_expected = bool(failed_attempts) and len(attempts) > 1
    hidden_or_fixture_workspace = any(
        marker in str(value).lower()
        for value in (request.workspace_root, run_dir)
        for marker in ("hidden_eval", "evals/hidden", "fixture")
    )
    return {
        "schema": "agent_repair_failure_diagnostics_v1",
        "final_failure_class": final_failure_class,
        "attempt_count": len(attempts),
        "repair_budget": retry_budget,
        "repair_budget_exhausted": bool(status == SOLVE_STATUS_FAILED and len(attempts) >= retry_budget + 1),
        "verifier_failure_type": _verifier_failure_type(final_failure_class),
        "all_attempts_failed_same_check": bool(check_names) and len(set(check_names)) == 1,
        "distinct_failure_signatures": sorted({str(item["stable_failure_signature"]) for item in failed_attempts}),
        "last_verifier_summary": last_failed.get("short_assertion_summary") if last_failed else None,
        "failure_is_model_or_patch_behavior": bool(failed_attempts) and not _infrastructure_failure_detected(final_failure_class, backend_status),
        "infrastructure_failure_detected": _infrastructure_failure_detected(final_failure_class, backend_status),
        "hidden_or_fixture_workspace_detected": hidden_or_fixture_workspace,
        "training_use_allowed": False,
        "no_winning_candidate": winning_attempt is None,
        "all_failed_edits_rolled_back": bool(failed_attempts) and all(not item["kept"] for item in failed_attempts),
        "per_attempt_evidence_paths_distinct": distinct_paths,
        "diagnostic_limitation": None if distinct_paths or not evidence_paths else "verifier evidence path reused",
        "repeated_failure_signature_detected": repeated_failure_signature,
        "repeated_failed_behavior_warning": (
            "same verifier failure signature repeated across repair attempts"
            if repeated_failure_signature
            else None
        ),
        "repair_prompt_includes_verifier_feedback": repair_prompt_expected,
        "repair_prompt_includes_failure_class": repair_prompt_expected and final_failure_class != "none",
        "attempts": attempt_diagnostics,
        "quality_claim": "none" if status != SOLVE_STATUS_VERIFIED else "verification_passed",
    }


def _backend_status_with_failure(backend_status: Dict, exc: BackendCandidateError) -> Dict:
    updated = dict(backend_status)
    updated["last_failure_class"] = exc.failure_class
    updated["last_failure_reason"] = exc.message
    if exc.details:
        updated["last_failure_details"] = dict(exc.details)
        for key in (
            "structured_contract_version",
            "generation_attempts",
            "malformed_retry_count",
            "structured_output_parse_status",
            "structured_candidate_valid",
            "raw_parse_status",
            "normalization_attempted",
            "normalization_applied",
            "normalization_kind",
            "normalization_rejected_reason",
            "final_parse_status",
            "final_schema_validation_status",
            "schema_validation_status",
            "unsafe_path_detected",
        ):
            if key in exc.details:
                updated[key] = exc.details[key]
    return updated


def _attempt_record(
    *,
    attempt_index: int,
    phase: str,
    origin: str,
    candidate: Candidate,
    files_touched: Sequence[str],
    verification: VerificationReport,
    critique: Optional[Critique],
    kept: bool,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_index=attempt_index,
        phase=phase,
        origin=origin,
        candidate_id=candidate.candidate_id,
        candidate_summary=candidate.summary,
        files_touched=tuple(files_touched),
        verification=verification,
        critique=critique,
        kept=kept,
    )


def _run_coding_attempt(
    *,
    context: ContextBundle,
    plan,
    run_dir: str,
    attempt_index: Optional[int] = None,
    phase: Optional[str] = None,
    candidate: Candidate,
) -> tuple[List[str], VerificationReport, bool]:
    session = WorkspaceEditSession(context.workspace_root)
    touched = session.apply_candidate(candidate)
    verifier_run_dir = run_dir
    if attempt_index is not None:
        safe_phase = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(phase or "attempt")).strip("_") or "attempt"
        verifier_run_dir = os.path.join(run_dir, "attempts", f"{attempt_index:02d}_{safe_phase}")
    verification = run_verification(
        workspace_root=context.workspace_root,
        checks=list(plan.checks),
        run_dir=verifier_run_dir,
    )
    if verification.overall_passed:
        session.commit()
        return touched, verification, True
    session.rollback()
    return touched, verification, False


def _generate_initial_candidate(
    *,
    backend: CodingModelBackend,
    request: TaskRequest,
    context: ContextBundle,
    plan,
) -> Candidate:
    return backend.generate_initial_candidate(request, context, plan)


def _generate_repair_candidate(
    *,
    backend: CodingModelBackend,
    request: TaskRequest,
    context: ContextBundle,
    plan,
    critique: Critique,
    previous_candidate: Candidate,
    attempt_history: Sequence[AttemptRecord],
) -> Optional[Candidate]:
    return backend.generate_repair_candidate(
        request,
        context,
        plan,
        critique,
        previous_candidate,
        attempt_history,
    )


def plan_task(request: TaskRequest) -> Dict:
    run_id, run_dir = _new_run_dir("plan")
    route = route_task(request)
    include_system_map = "architecture" in request.task_text.lower() or "system" in request.task_text.lower()
    retrieval_notes = ()
    if route.route == ROUTE_CODING:
        retrieval_notes = tuple(
            summarize_for_context(
                query=request.task_text,
                file_hints=request.file_hints,
            )
        )
    context = build_context(
        request,
        max_file_chars=agent_cfg.max_file_excerpt_chars,
        include_system_map=include_system_map,
        extra_notes=retrieval_notes,
    )
    plan = build_plan(request, route, context)
    return _finalize(
        run_id=run_id,
        run_dir=run_dir,
        operation="plan",
        request=request,
        route=route,
        backend_status={"configured": False, "available": False, "kind": "none", "reason": "plan operation does not require backend"},
        plan=plan,
        context=context,
        verification=_empty_verification("plan operation does not verify workspace state"),
        status=PLAN_STATUS_READY if route.supported else SOLVE_STATUS_UNSUPPORTED,
        files_touched=[],
        blocked_reason=None if route.supported else route.reason,
        attempts=_empty_attempt_history(),
    )


def verify_task(request: TaskRequest) -> Dict:
    run_id, run_dir = _new_run_dir("verify")
    route = route_task(request)
    include_system_map = "architecture" in request.task_text.lower() or "system" in request.task_text.lower()
    retrieval_notes = ()
    if route.route == ROUTE_CODING:
        retrieval_notes = tuple(
            summarize_for_context(
                query=request.task_text,
                file_hints=request.file_hints,
            )
        )
    context = build_context(
        request,
        max_file_chars=agent_cfg.max_file_excerpt_chars,
        include_system_map=include_system_map,
        extra_notes=retrieval_notes,
    )
    plan = build_plan(request, route, context)

    if route.route == ROUTE_EXACT:
        try:
            exact = execute_exact_task(request.task_text, context.workspace_root)
        except (OSError, ValueError) as exc:
            return _finalize(
                run_id=run_id,
                run_dir=run_dir,
                operation="verify",
                request=request,
                route=route,
                backend_status={"configured": False, "available": False, "kind": "deterministic_exact_tool"},
                plan=plan,
                context=context,
                verification=_empty_verification("deterministic exact-task handler unavailable"),
                status=SOLVE_STATUS_UNSUPPORTED,
                files_touched=[],
                blocked_reason=str(exc),
                attempts=_empty_attempt_history(),
                final_origin="deterministic_exact_tool",
            )
        verification = run_verification(
            workspace_root=context.workspace_root,
            checks=list(plan.checks),
            run_dir=run_dir,
            exact_result=exact,
        )
        status = SOLVE_STATUS_VERIFIED if verification.overall_passed else SOLVE_STATUS_FAILED
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="verify",
            request=request,
            route=route,
            backend_status={"configured": False, "available": False, "kind": "deterministic_exact_tool"},
            plan=plan,
            context=context,
            verification=verification,
            status=status,
            files_touched=[],
            blocked_reason=None if status == SOLVE_STATUS_VERIFIED else "exact validation failed",
            attempts=_empty_attempt_history(),
            final_origin="deterministic_exact_tool",
        )

    if route.route != ROUTE_CODING:
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="verify",
            request=request,
            route=route,
            backend_status={"configured": False, "available": False, "kind": "none"},
            plan=plan,
            context=context,
            verification=_empty_verification("verification route unsupported for the current agent command"),
            status=SOLVE_STATUS_UNSUPPORTED,
            files_touched=[],
            blocked_reason=route.reason,
            attempts=_empty_attempt_history(),
        )

    if not plan.checks:
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="verify",
            request=request,
            route=route,
            backend_status={"configured": False, "available": False, "kind": "none"},
            plan=plan,
            context=context,
            verification=_empty_verification("no meaningful validator exists for this coding task"),
            status=SOLVE_STATUS_BLOCKED,
            files_touched=[],
            blocked_reason="no meaningful validator exists",
            retry_budget=plan.retry_budget,
            attempts=_empty_attempt_history(),
        )

    verification = run_verification(
        workspace_root=context.workspace_root,
        checks=list(plan.checks),
        run_dir=run_dir,
    )
    status = SOLVE_STATUS_VERIFIED if verification.overall_passed else SOLVE_STATUS_FAILED
    return _finalize(
        run_id=run_id,
        run_dir=run_dir,
        operation="verify",
        request=request,
        route=route,
        backend_status={"configured": False, "available": False, "kind": "none"},
        plan=plan,
        context=context,
        verification=verification,
        status=status,
        files_touched=[],
        blocked_reason=None if status == SOLVE_STATUS_VERIFIED else "workspace verification failed",
        retry_budget=plan.retry_budget,
        attempts=_empty_attempt_history(),
        final_origin="verification_only",
    )


def solve_task(
    request: TaskRequest,
    *,
    backend_script_path: Optional[str] = None,
    optimize: bool = False,
) -> Dict:
    run_id, run_dir = _new_run_dir("solve")
    route = route_task(request)
    include_system_map = "architecture" in request.task_text.lower() or "system" in request.task_text.lower()
    retrieval_notes = ()
    if route.route == ROUTE_CODING:
        retrieval_notes = tuple(
            summarize_for_context(
                query=request.task_text,
                file_hints=request.file_hints,
            )
        )
    context = build_context(
        request,
        max_file_chars=agent_cfg.max_file_excerpt_chars,
        include_system_map=include_system_map,
        extra_notes=retrieval_notes,
    )
    plan = build_plan(request, route, context)

    if route.route == ROUTE_EXACT:
        try:
            exact = execute_exact_task(request.task_text, context.workspace_root)
        except (OSError, ValueError) as exc:
            return _finalize(
                run_id=run_id,
                run_dir=run_dir,
                operation="solve",
                request=request,
                route=route,
                backend_status={"configured": False, "available": False, "kind": "deterministic_exact_tool"},
                plan=plan,
                context=context,
                verification=_empty_verification("deterministic exact-task handler unavailable"),
                status=SOLVE_STATUS_UNSUPPORTED,
                files_touched=[],
                blocked_reason=str(exc),
                attempts=_empty_attempt_history(),
                final_origin="deterministic_exact_tool",
            )
        verification = run_verification(
            workspace_root=context.workspace_root,
            checks=list(plan.checks),
            run_dir=run_dir,
            exact_result=exact,
        )
        status = SOLVE_STATUS_VERIFIED if verification.overall_passed else SOLVE_STATUS_FAILED
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="solve",
            request=request,
            route=route,
            backend_status={"configured": False, "available": False, "kind": "deterministic_exact_tool"},
            plan=plan,
            context=context,
            verification=verification,
            status=status,
            files_touched=[],
            blocked_reason=None if status == SOLVE_STATUS_VERIFIED else "exact validation failed",
            attempts=_empty_attempt_history(),
            final_origin="deterministic_exact_tool",
        )

    if route.route != ROUTE_CODING:
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="solve",
            request=request,
            route=route,
            backend_status={"configured": False, "available": False, "kind": "none"},
            plan=plan,
            context=context,
            verification=_empty_verification("route unsupported for the current agent solve path"),
            status=SOLVE_STATUS_UNSUPPORTED,
            files_touched=[],
            blocked_reason=route.reason,
            attempts=_empty_attempt_history(),
        )

    if not plan.checks:
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="solve",
            request=request,
            route=route,
            backend_status={"configured": False, "available": False, "kind": "none"},
            plan=plan,
            context=context,
            verification=_empty_verification("no meaningful validator exists for this coding task"),
            status=SOLVE_STATUS_BLOCKED,
            files_touched=[],
            blocked_reason="no meaningful validator exists",
            retry_budget=plan.retry_budget,
            attempts=_empty_attempt_history(),
        )

    backend, backend_status = load_backend(
        workspace_root=context.workspace_root,
        backend_script_path=backend_script_path,
    )
    if backend is None:
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="solve",
            request=request,
            route=route,
            backend_status=backend_status,
            plan=plan,
            context=context,
            verification=_empty_verification("coding backend unavailable; solve must fail closed"),
            status=SOLVE_STATUS_BLOCKED,
            files_touched=[],
            blocked_reason=backend_status.get("reason", "no backend available"),
            retry_budget=plan.retry_budget,
            attempts=_empty_attempt_history(),
        )
    attempts: List[AttemptRecord] = []
    retry_budget = max(0, int(plan.retry_budget))
    winning_candidate: Optional[Candidate] = None
    final_origin = "none"
    winning_attempt: Optional[int] = None
    optimization_status = OPTIMIZATION_NOT_ATTEMPTED
    files_touched: List[str] = []

    try:
        current_candidate = _generate_initial_candidate(
            backend=backend,
            request=request,
            context=context,
            plan=plan,
        )
    except BackendCandidateError as exc:
        backend_status = _backend_status_with_failure(backend_status, exc)
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="solve",
            request=request,
            route=route,
            backend_status=backend_status,
            plan=plan,
            context=context,
            verification=_empty_verification("backend could not produce an initial candidate"),
            status=SOLVE_STATUS_BLOCKED,
            files_touched=[],
            blocked_reason=f"backend initial candidate generation failed: {exc.failure_class}: {exc.message}",
            retry_budget=retry_budget,
            attempts=tuple(attempts),
        )
    except (OSError, ValueError, KeyError) as exc:
        return _finalize(
            run_id=run_id,
            run_dir=run_dir,
            operation="solve",
            request=request,
            route=route,
            backend_status=backend_status,
            plan=plan,
            context=context,
            verification=_empty_verification("backend could not produce an initial candidate"),
            status=SOLVE_STATUS_BLOCKED,
            files_touched=[],
            blocked_reason=f"backend initial candidate generation failed: {exc}",
            retry_budget=retry_budget,
            attempts=tuple(attempts),
        )

    verification = _empty_verification("candidate verification did not run")
    status = SOLVE_STATUS_FAILED
    blocked_reason = "verification failed"

    for repair_index in range(retry_budget + 1):
        phase = "initial" if repair_index == 0 else "repair"
        origin = "initial" if repair_index == 0 else "repaired"
        touched, verification, passed = _run_coding_attempt(
            context=context,
            plan=plan,
            run_dir=run_dir,
            attempt_index=repair_index + 1,
            phase=phase,
            candidate=current_candidate,
        )
        files_touched = list(touched)
        if passed:
            record = _attempt_record(
                attempt_index=repair_index + 1,
                phase=phase,
                origin=origin,
                candidate=current_candidate,
                files_touched=touched,
                verification=verification,
                critique=None,
                kept=True,
            )
            attempts.append(record)
            winning_candidate = current_candidate
            final_origin = origin
            winning_attempt = repair_index + 1
            status = SOLVE_STATUS_VERIFIED
            blocked_reason = None
            break

        critique = critique_failure(
            verification=verification,
            candidate=current_candidate,
            plan=plan,
            context=context,
        )
        attempts.append(
            _attempt_record(
                attempt_index=repair_index + 1,
                phase=phase,
                origin=origin,
                candidate=current_candidate,
                files_touched=touched,
                verification=verification,
                critique=critique,
                kept=False,
            )
        )
        blocked_reason = critique.blocked_reason or "verification failed; applied edits were rolled back"

        if critique.blocked_reason:
            status = SOLVE_STATUS_BLOCKED
            break
        if repair_index >= retry_budget:
            status = SOLVE_STATUS_FAILED
            break

        repair_critique = _critique_with_repair_feedback(
            critique=critique,
            previous_candidate=current_candidate,
            attempt_history=tuple(attempts),
            plan_retry_budget=retry_budget,
        )
        try:
            next_candidate = _generate_repair_candidate(
                backend=backend,
                request=request,
                context=context,
                plan=plan,
                critique=repair_critique,
                previous_candidate=current_candidate,
                attempt_history=tuple(attempts),
            )
        except BackendCandidateError as exc:
            backend_status = _backend_status_with_failure(backend_status, exc)
            status = SOLVE_STATUS_BLOCKED
            blocked_reason = f"backend repair candidate generation failed: {exc.failure_class}: {exc.message}"
            break
        except (OSError, ValueError, KeyError) as exc:
            status = SOLVE_STATUS_BLOCKED
            blocked_reason = f"backend repair candidate generation failed: {exc}"
            break
        if next_candidate is None:
            status = SOLVE_STATUS_FAILED
            blocked_reason = "verification failed; applied edits were rolled back and no additional repair candidate was available"
            break
        current_candidate = next_candidate

    if status == SOLVE_STATUS_VERIFIED and optimize and winning_candidate is not None:
        try:
            optimization_status, optimization_attempt, optimized_candidate, optimization_touched = run_optimizer(
                request=request,
                context=context,
                plan=plan,
                backend=backend,
                winning_candidate=winning_candidate,
                attempt_history=tuple(attempts),
                run_dir=run_dir,
            )
        except BackendCandidateError as exc:
            backend_status = _backend_status_with_failure(backend_status, exc)
            optimization_status = "blocked_unverified"
            optimization_attempt = None
            optimized_candidate = None
            optimization_touched = []
        if optimization_attempt is not None:
            attempts.append(optimization_attempt)
        if optimized_candidate is not None and optimization_attempt is not None:
            winning_candidate = optimized_candidate
            final_origin = "optimized"
            winning_attempt = optimization_attempt.attempt_index
            files_touched = list(optimization_touched)
            verification = optimization_attempt.verification
        elif optimization_attempt is not None and optimization_attempt.verification is not None:
            verification = attempts[winning_attempt - 1].verification if winning_attempt is not None else verification

    return _finalize(
        run_id=run_id,
        run_dir=run_dir,
        operation="solve",
        request=request,
        route=route,
        backend_status=backend_status,
        plan=plan,
        context=context,
        verification=verification,
        status=status,
        files_touched=files_touched if status == SOLVE_STATUS_VERIFIED else [],
        blocked_reason=blocked_reason,
        candidate=winning_candidate,
        retry_budget=retry_budget,
        attempts=tuple(attempts),
        final_origin=final_origin,
        winning_attempt=winning_attempt,
        optimization_status=optimization_status,
    )
