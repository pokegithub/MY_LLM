"""Conservative task router for the Phase 1 verified coding agent."""

from __future__ import annotations

import re

from agent.types import (
    ROUTE_CODING,
    ROUTE_EXACT,
    ROUTE_FACTUAL,
    ROUTE_OPEN,
    ROUTE_UNSUPPORTED,
    RouteDecision,
    TaskRequest,
)


_EXACT_PATTERNS = [
    re.compile(r"^\s*arithmetic\s*:\s*.+$", re.IGNORECASE),
    re.compile(r'^\s*count\s+(substring|char|character|letter)\s+".+"\s+in\s+".+"\s*$', re.IGNORECASE),
    re.compile(r'^\s*contains\s+".+"\s+in\s+".+"\s*$', re.IGNORECASE),
    re.compile(r'^\s*index\s+of\s+".+"\s+in\s+".+"\s*$', re.IGNORECASE),
    re.compile(r"^\s*json\s+(validate|format)\s+.+$", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*file\s+exists\s+.+$", re.IGNORECASE),
    re.compile(r'^\s*symbol\s+exists\s+".+"\s+in\s+.+$', re.IGNORECASE),
]

_CODING_CUES = (
    "fix ",
    "implement ",
    "edit ",
    "update ",
    "refactor ",
    "patch ",
    "change ",
    "rename ",
    "add test",
    "write code",
)

_FACTUAL_CUES = (
    "latest ",
    "what is ",
    "who is ",
    "when did ",
    "according to ",
)

_OPEN_CUES = (
    "write a poem",
    "brainstorm",
    "draft copy",
    "write a story",
)


def route_task(request: TaskRequest) -> RouteDecision:
    text = request.task_text.strip()
    lowered = text.lower()

    if request.task_type_hint:
        hint = request.task_type_hint.strip().lower()
        if hint in {ROUTE_EXACT, ROUTE_CODING, ROUTE_FACTUAL, ROUTE_OPEN, ROUTE_UNSUPPORTED}:
            deterministic = hint == ROUTE_EXACT
            supported = hint in {ROUTE_EXACT, ROUTE_CODING}
            return RouteDecision(
                route=hint,
                reason=f"explicit task_type_hint={hint}",
                deterministic_required=deterministic,
                supported=supported,
            )

    if any(pattern.match(text) for pattern in _EXACT_PATTERNS):
        return RouteDecision(
            route=ROUTE_EXACT,
            reason="matched deterministic exact-task grammar",
            deterministic_required=True,
            supported=True,
        )

    if request.file_hints or any(cue in lowered for cue in _CODING_CUES):
        return RouteDecision(
            route=ROUTE_CODING,
            reason="task requests code edits or file-scoped changes",
            deterministic_required=False,
            supported=True,
        )

    if any(cue in lowered for cue in _FACTUAL_CUES):
        return RouteDecision(
            route=ROUTE_FACTUAL,
            reason="task appears factual/retrieval-oriented",
            deterministic_required=False,
            supported=False,
        )

    if any(cue in lowered for cue in _OPEN_CUES):
        return RouteDecision(
            route=ROUTE_OPEN,
            reason="task appears open-ended generative",
            deterministic_required=False,
            supported=False,
        )

    return RouteDecision(
        route=ROUTE_UNSUPPORTED,
        reason="task does not match a supported deterministic or coding-edit path",
        deterministic_required=False,
        supported=False,
    )
