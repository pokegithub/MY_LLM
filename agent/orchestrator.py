"""Top-level Phase 1 agent orchestration."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional

from agent.backend import load_backend
from agent.context import build_context
from agent.exact_tools import execute_exact_task
from agent.planner import build_plan
from agent.router import route_task
from agent.types import (
    PLAN_STATUS_READY,
    ROUTE_CODING,
    ROUTE_EXACT,
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
    )
    payload = to_dict(result)
    payload["schema"] = "agent_phase1_report_v1"
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
    _write_report(run_dir, payload)
    return payload


def plan_task(request: TaskRequest) -> Dict:
    run_id, run_dir = _new_run_dir("plan")
    route = route_task(request)
    include_system_map = "architecture" in request.task_text.lower() or "system" in request.task_text.lower()
    context = build_context(
        request,
        max_file_chars=agent_cfg.max_file_excerpt_chars,
        include_system_map=include_system_map,
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
    )


def verify_task(request: TaskRequest) -> Dict:
    run_id, run_dir = _new_run_dir("verify")
    route = route_task(request)
    include_system_map = "architecture" in request.task_text.lower() or "system" in request.task_text.lower()
    context = build_context(
        request,
        max_file_chars=agent_cfg.max_file_excerpt_chars,
        include_system_map=include_system_map,
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
            verification=_empty_verification("verification route unsupported in Phase 1"),
            status=SOLVE_STATUS_UNSUPPORTED,
            files_touched=[],
            blocked_reason=route.reason,
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
    )


def solve_task(
    request: TaskRequest,
    *,
    backend_script_path: Optional[str] = None,
) -> Dict:
    run_id, run_dir = _new_run_dir("solve")
    route = route_task(request)
    include_system_map = "architecture" in request.task_text.lower() or "system" in request.task_text.lower()
    context = build_context(
        request,
        max_file_chars=agent_cfg.max_file_excerpt_chars,
        include_system_map=include_system_map,
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
            verification=_empty_verification("route unsupported in Phase 1"),
            status=SOLVE_STATUS_UNSUPPORTED,
            files_touched=[],
            blocked_reason=route.reason,
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
        )

    candidate = backend.generate_candidate(request, context, plan)
    session = WorkspaceEditSession(context.workspace_root)
    touched = session.apply_candidate(candidate)
    verification = run_verification(
        workspace_root=context.workspace_root,
        checks=list(plan.checks),
        run_dir=run_dir,
    )
    if verification.overall_passed:
        session.commit()
        status = SOLVE_STATUS_VERIFIED
        blocked_reason = None
    else:
        session.rollback()
        status = SOLVE_STATUS_FAILED
        blocked_reason = "verification failed; applied edits were rolled back"

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
        files_touched=touched,
        blocked_reason=blocked_reason,
        candidate=candidate,
    )
