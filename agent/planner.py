"""Machine-readable planner for the verified coding agent."""

from __future__ import annotations

from typing import List

from agent.types import (
    ROUTE_CODING,
    ROUTE_EXACT,
    ContextBundle,
    PlanCheck,
    RouteDecision,
    SolvePlan,
    TaskRequest,
)
from config import agent_cfg


def _normalize_check(raw: str) -> PlanCheck | None:
    text = raw.strip()
    if text.startswith("compileall:"):
        return PlanCheck("compileall", text.split(":", 1)[1].strip())
    if text.startswith("pytest:"):
        return PlanCheck("pytest", text.split(":", 1)[1].strip())
    if text.startswith("unittest:"):
        return PlanCheck("unittest", text.split(":", 1)[1].strip())
    if text.startswith("path_exists:"):
        return PlanCheck("path_exists", text.split(":", 1)[1].strip())
    return None


def build_plan(
    request: TaskRequest,
    route: RouteDecision,
    context: ContextBundle,
) -> SolvePlan:
    checks: List[PlanCheck] = []
    target_files = [item.path for item in context.files if item.note.startswith("explicit")]

    for raw in request.checks:
        check = _normalize_check(raw)
        if check is not None:
            checks.append(check)

    if route.route == ROUTE_EXACT:
        checks.append(PlanCheck("exact_validation", "rerun deterministic handler"))
        return SolvePlan(
            route=route.route,
            subgoals=("parse deterministic task", "compute exact result", "revalidate exact result"),
            target_files=tuple(target_files),
            checks=tuple(checks),
            risk_level="low",
            success_criteria=("deterministic handler matched", "exact validation passed"),
            retry_budget=0,
            stop_conditions=(
                "stop if no deterministic handler matches",
                "stop if exact validation fails",
            ),
        )

    if route.route == ROUTE_CODING:
        if not checks:
            python_targets = [
                item.path for item in context.files
                if item.exists and item.path.endswith(".py") and "explicit file hint" in item.note
            ]
            if python_targets:
                checks.append(PlanCheck("compileall", ",".join(python_targets)))
            related_tests = [
                item.path for item in context.files
                if item.exists and "related test" in item.note and item.path.endswith(".py")
            ]
            if related_tests:
                checks.append(PlanCheck("pytest", " ".join(related_tests) + " -q"))

        risk = "low" if len(target_files) <= 1 else "medium"
        if len(checks) >= 3:
            risk = "high"
        return SolvePlan(
            route=route.route,
            subgoals=(
                "inspect targeted files",
                "generate one candidate patch",
                "apply patch in reversible workspace session",
                "run planned verification checks",
            ),
            target_files=tuple(target_files),
            checks=tuple(checks),
            risk_level=risk,
            success_criteria=(
                "at least one meaningful verification check exists",
                "all verification checks pass",
                "failed candidate edits are rolled back",
            ),
            retry_budget=max(0, int(agent_cfg.default_retry_budget)),
            stop_conditions=(
                "stop if no backend is configured",
                "stop if no meaningful validator exists",
                "stop after retry budget is exhausted",
                "stop if the critic marks repair as unjustified",
            ),
        )

    return SolvePlan(
        route=route.route,
        subgoals=("classify task conservatively",),
        target_files=tuple(target_files),
        checks=tuple(checks),
        risk_level="low",
        success_criteria=("task is either supported or explicitly blocked",),
        retry_budget=0,
        stop_conditions=("stop because this route is not implemented in the current agent phase",),
    )
