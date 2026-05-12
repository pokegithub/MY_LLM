"""Top-level orchestration for the verified coding agent."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional, Sequence

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


def _backend_status_with_failure(backend_status: Dict, exc: BackendCandidateError) -> Dict:
    updated = dict(backend_status)
    updated["last_failure_class"] = exc.failure_class
    updated["last_failure_reason"] = exc.message
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
    candidate: Candidate,
) -> tuple[List[str], VerificationReport, bool]:
    session = WorkspaceEditSession(context.workspace_root)
    touched = session.apply_candidate(candidate)
    verification = run_verification(
        workspace_root=context.workspace_root,
        checks=list(plan.checks),
        run_dir=run_dir,
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

        try:
            next_candidate = _generate_repair_candidate(
                backend=backend,
                request=request,
                context=context,
                plan=plan,
                critique=critique,
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
