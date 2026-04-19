"""Structured security-event telemetry for policy and integrity paths."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from observability.metrics import MetricsLogger


_EVENT_LOGGER: Optional[MetricsLogger] = None
_EVENT_PATH: Optional[str] = None


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _security_events_enabled() -> bool:
    return _is_truthy(os.environ.get("MYLLM_SECURITY_EVENTS_ENABLED", "1"))


def _security_events_path() -> str:
    configured = os.environ.get("MYLLM_SECURITY_EVENTS_PATH", "").strip()
    if configured:
        return configured
    return os.path.join(".", "run_artifacts", "security_events.jsonl")


def _get_logger() -> MetricsLogger:
    global _EVENT_LOGGER
    global _EVENT_PATH

    path = _security_events_path()
    if _EVENT_LOGGER is None or _EVENT_PATH != path:
        _EVENT_LOGGER = MetricsLogger(path)
        _EVENT_PATH = path
    return _EVENT_LOGGER


def log_security_event(event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Emit a structured security event; never raises to caller."""
    if not _security_events_enabled():
        return

    item: Dict[str, Any] = {
        "kind": "security_event",
        "event_type": str(event_type),
    }
    if payload:
        item.update(payload)

    try:
        _get_logger().log(item)
    except Exception:
        # Security telemetry must not break primary execution paths.
        return
