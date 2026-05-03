"""Post-green optimizer for the verified coding agent."""

from __future__ import annotations

from typing import Optional, Sequence

from agent.backend import CodingModelBackend
from agent.critic import critique_failure
from agent.types import (
    AttemptRecord,
    Candidate,
    ContextBundle,
    OPTIMIZATION_BLOCKED,
    OPTIMIZATION_KEPT,
    OPTIMIZATION_ROLLED_BACK,
    SolvePlan,
    TaskRequest,
)
from agent.verify import run_verification
from agent.workspace import WorkspaceEditSession


def run_optimizer(
    *,
    request: TaskRequest,
    context: ContextBundle,
    plan: SolvePlan,
    backend: CodingModelBackend,
    winning_candidate: Candidate,
    attempt_history: Sequence[AttemptRecord],
    run_dir: str,
) -> tuple[str, Optional[AttemptRecord], Optional[Candidate], list[str]]:
    optimization_candidate = backend.generate_optimization_candidate(
        request,
        context,
        plan,
        winning_candidate,
        attempt_history,
    )
    if optimization_candidate is None:
        return OPTIMIZATION_BLOCKED, None, None, []

    session = WorkspaceEditSession(context.workspace_root)
    touched = session.apply_candidate(optimization_candidate)
    verification = run_verification(
        workspace_root=context.workspace_root,
        checks=list(plan.checks),
        run_dir=run_dir,
    )
    if verification.overall_passed:
        session.commit()
        return (
            OPTIMIZATION_KEPT,
            AttemptRecord(
                attempt_index=len(attempt_history) + 1,
                phase="optimization",
                origin="optimized",
                candidate_id=optimization_candidate.candidate_id,
                candidate_summary=optimization_candidate.summary,
                files_touched=tuple(touched),
                verification=verification,
                critique=None,
                kept=True,
            ),
            optimization_candidate,
            touched,
        )

    session.rollback()
    critique = critique_failure(
        verification=verification,
        candidate=optimization_candidate,
        plan=plan,
        context=context,
    )
    return (
        OPTIMIZATION_ROLLED_BACK,
        AttemptRecord(
            attempt_index=len(attempt_history) + 1,
            phase="optimization",
            origin="optimized",
            candidate_id=optimization_candidate.candidate_id,
            candidate_summary=optimization_candidate.summary,
            files_touched=tuple(touched),
            verification=verification,
            critique=critique,
            kept=False,
        ),
        None,
        touched,
    )
