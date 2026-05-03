"""Observability package with lazy training telemetry imports."""

from observability.metrics import MetricsLogger

__all__ = [
    "MetricsLogger",
    "PerformanceMonitor",
    "TrainingTelemetry",
    "default_telemetry_path",
]


def __getattr__(name: str):
    if name in {"PerformanceMonitor", "TrainingTelemetry", "default_telemetry_path"}:
        from observability.training_telemetry import (
            PerformanceMonitor,
            TrainingTelemetry,
            default_telemetry_path,
        )

        exports = {
            "PerformanceMonitor": PerformanceMonitor,
            "TrainingTelemetry": TrainingTelemetry,
            "default_telemetry_path": default_telemetry_path,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
