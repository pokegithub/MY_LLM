"""Structured contracts for the Phase 1 verified coding agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional, Tuple


ROUTE_EXACT = "exact_symbolic"
ROUTE_CODING = "coding_edit"
ROUTE_FACTUAL = "factual_retrieval"
ROUTE_OPEN = "open_generation"
ROUTE_UNSUPPORTED = "unsupported"

SOLVE_STATUS_VERIFIED = "verified_success"
SOLVE_STATUS_FAILED = "verification_failed"
SOLVE_STATUS_BLOCKED = "blocked_unverified"
SOLVE_STATUS_UNSUPPORTED = "unsupported"
PLAN_STATUS_READY = "plan_ready"


@dataclass(frozen=True)
class TaskRequest:
    task_text: str
    file_hints: Tuple[str, ...] = ()
    checks: Tuple[str, ...] = ()
    workspace_root: str = "."
    task_file: Optional[str] = None
    task_type_hint: Optional[str] = None


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    deterministic_required: bool
    supported: bool


@dataclass(frozen=True)
class PlanCheck:
    check_type: str
    spec: str
    meaningful: bool = True


@dataclass(frozen=True)
class SolvePlan:
    route: str
    subgoals: Tuple[str, ...]
    target_files: Tuple[str, ...]
    checks: Tuple[PlanCheck, ...]
    risk_level: str
    success_criteria: Tuple[str, ...]
    retry_budget: int
    stop_conditions: Tuple[str, ...]


@dataclass(frozen=True)
class ContextFile:
    path: str
    exists: bool
    included: bool
    size_bytes: int
    content_excerpt: str = ""
    note: str = ""


@dataclass(frozen=True)
class ContextBundle:
    workspace_root: str
    files_considered: Tuple[str, ...]
    files: Tuple[ContextFile, ...]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FileEdit:
    path: str
    new_content: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    summary: str
    edits: Tuple[FileEdit, ...]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    check_type: str
    spec: str
    passed: bool
    exit_code: Optional[int]
    summary: str
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    overall_passed: bool
    meaningful: bool
    checks: Tuple[CheckResult, ...]
    summary: str
    quality_claim: str


@dataclass(frozen=True)
class SolveResult:
    run_id: str
    operation: str
    request: TaskRequest
    route: RouteDecision
    backend_status: Dict[str, Any]
    plan: SolvePlan
    context: ContextBundle
    verification: VerificationReport
    status: str
    files_touched: Tuple[str, ...]
    blocked_reason: Optional[str]
    quality_claim: str
    report_path: str
    candidate: Optional[Candidate] = None


def to_dict(value: Any) -> Any:
    """Recursively normalize dataclasses for JSON reports."""
    if is_dataclass(value):
        return {k: to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value
