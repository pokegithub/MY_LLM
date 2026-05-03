"""Evidence-oriented failure classification for the coding agent."""

from __future__ import annotations

from typing import Iterable, Optional

from agent.types import (
    FAILURE_ASSERTION,
    FAILURE_BEHAVIOR,
    FAILURE_IMPORT,
    FAILURE_PERFORMANCE,
    FAILURE_RUNTIME,
    FAILURE_SYNTAX,
    FAILURE_TIMEOUT,
    FAILURE_UNSUPPORTED,
    FAILURE_UNVERIFIED,
    FAILURE_VERIFIER_MISSING,
    Candidate,
    ContextBundle,
    Critique,
    SolvePlan,
    VerificationReport,
)


def _target_files(plan: SolvePlan, context: ContextBundle, candidate: Optional[Candidate]) -> tuple[str, ...]:
    targets = list(plan.target_files)
    if not targets and candidate is not None:
        targets.extend(edit.path for edit in candidate.edits)
    if not targets:
        targets.extend(item.path for item in context.files if item.included)
    deduped: list[str] = []
    for target in targets:
        if target not in deduped:
            deduped.append(target)
    return tuple(deduped)


def _contains_any(text: str, fragments: Iterable[str]) -> bool:
    return any(fragment in text for fragment in fragments)


def critique_failure(
    *,
    verification: VerificationReport,
    candidate: Optional[Candidate],
    plan: SolvePlan,
    context: ContextBundle,
) -> Critique:
    targets = _target_files(plan, context, candidate)

    if not verification.meaningful or not verification.checks:
        return Critique(
            failure_class=FAILURE_VERIFIER_MISSING,
            root_cause="no meaningful verifier evidence was available",
            repair_targets=targets,
            blocked_reason="repair is unjustified without a meaningful verifier",
            confidence=1.0,
            evidence_summary=verification.summary,
        )

    failed_checks = [check for check in verification.checks if not check.passed]
    first_failed = failed_checks[0] if failed_checks else verification.checks[0]
    combined = " ".join(
        filter(
            None,
            [
                first_failed.summary,
                str(first_failed.evidence.get("command", "")),
                verification.summary,
            ],
        )
    ).lower()

    if _contains_any(combined, ("timed out", "timeout")):
        return Critique(
            failure_class=FAILURE_TIMEOUT,
            root_cause="verification command timed out",
            repair_targets=targets,
            blocked_reason=None,
            confidence=0.98,
            evidence_summary=first_failed.summary,
        )

    if _contains_any(combined, ("syntaxerror", "invalid syntax", "expected ':'", "eol while scanning")):
        return Critique(
            failure_class=FAILURE_SYNTAX,
            root_cause="candidate introduced invalid Python syntax",
            repair_targets=targets,
            blocked_reason=None,
            confidence=0.98,
            evidence_summary=first_failed.summary,
        )

    if _contains_any(combined, ("modulenotfounderror", "importerror", "cannot import name")):
        return Critique(
            failure_class=FAILURE_IMPORT,
            root_cause="candidate broke module import behavior",
            repair_targets=targets,
            blocked_reason=None,
            confidence=0.96,
            evidence_summary=first_failed.summary,
        )

    if _contains_any(combined, ("unsupported verification check type",)):
        return Critique(
            failure_class=FAILURE_UNSUPPORTED,
            root_cause="verification contract includes an unsupported check type",
            repair_targets=targets,
            blocked_reason="repair is unjustified until the verifier contract is supported",
            confidence=1.0,
            evidence_summary=first_failed.summary,
        )

    if _contains_any(combined, ("assertionerror", "assert ", "expected", "!= ", "== ", "failed")):
        return Critique(
            failure_class=FAILURE_ASSERTION,
            root_cause="candidate did not satisfy the asserted behavior",
            repair_targets=targets,
            blocked_reason=None,
            confidence=0.9,
            evidence_summary=first_failed.summary,
        )

    if _contains_any(combined, ("performance", "too slow", "regression")):
        return Critique(
            failure_class=FAILURE_PERFORMANCE,
            root_cause="verification detected a performance-oriented regression",
            repair_targets=targets,
            blocked_reason=None,
            confidence=0.85,
            evidence_summary=first_failed.summary,
        )

    if _contains_any(combined, ("traceback", "exception", "error", "typeerror", "valueerror", "attributeerror")):
        return Critique(
            failure_class=FAILURE_RUNTIME,
            root_cause="candidate triggered a runtime failure",
            repair_targets=targets,
            blocked_reason=None,
            confidence=0.88,
            evidence_summary=first_failed.summary,
        )

    if failed_checks:
        return Critique(
            failure_class=FAILURE_BEHAVIOR,
            root_cause="verification failed without a clearer syntax or import signature",
            repair_targets=targets,
            blocked_reason=None,
            confidence=0.72,
            evidence_summary=first_failed.summary,
        )

    return Critique(
        failure_class=FAILURE_UNVERIFIED,
        root_cause="verification result was inconclusive",
        repair_targets=targets,
        blocked_reason="repair is unjustified because failure evidence is inconclusive",
        confidence=0.5,
        evidence_summary=verification.summary,
    )
