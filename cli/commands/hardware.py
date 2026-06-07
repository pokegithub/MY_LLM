"""Hardware validation CLI command wrappers."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from cli.output import print_banner


def _default_emit_json(_payload: dict[str, Any]) -> None:
    raise RuntimeError("emit_json callback is required for JSON output")


def _default_write_json_report(
    _payload: dict[str, Any],
    _path: str,
) -> str:
    raise RuntimeError("write_json_report callback is required")


def run_hardware_validate(
    output_json: bool = False,
    emit_json: Callable[[dict[str, Any]], None] | None = None,
    write_json_report: Callable[[dict[str, Any], str], str] | None = None,
    report_path: str | None = None,
    default_report_path: str = "./run_artifacts/hardware_readiness_report.json",
) -> None:
    """Report current-machine hardware validation evidence."""
    import train

    report = train.run_hardware_readiness_validation()
    resolved_report_path = (write_json_report or _default_write_json_report)(
        report,
        report_path or default_report_path,
    )
    report["report_path"] = resolved_report_path
    if output_json:
        (emit_json or _default_emit_json)(report)
        if not report["ok"]:
            sys.exit(1)
        return

    print_banner("HARDWARE VALIDATION")
    nvidia = report.get("nvidia_smi", {})
    system_gpus = nvidia.get("gpus", []) if isinstance(nvidia, dict) else []
    print(f"  cuda_available        : {report['cuda_available']}")
    print(f"  cuda_device_count     : {report['cuda_device_count']}")
    print(f"  torch_version         : {report['torch_version']}")
    print(f"  torch_cuda_runtime    : {report['torch_cuda_runtime']}")
    print(f"  cudnn_available       : {report['cudnn_available']}")
    print(f"  system_gpu_detected   : {report['system_gpu_detected']}")
    if system_gpus:
        first_gpu = system_gpus[0]
        print(f"  system_gpu_name       : {first_gpu.get('name', 'unknown')}")
        print(f"  system_gpu_vram_gb    : {first_gpu.get('total_memory_gb', 'unknown')}")
        print(f"  system_gpu_compute    : {first_gpu.get('compute_capability', 'unknown')}")
    print(f"  gpu_classification    : {report['gpu_result_classification']}")
    print(f"  evidence_level        : {report['evidence_level']}")
    print(f"  hardware_ready        : {report['hardware_ready_for_training']}")
    tensor_ops = report.get("gpu_tensor_ops", {})
    print(f"  gpu_tensor_op         : {tensor_ops.get('status', 'unknown')}")
    validation = report.get("validation") or {}
    print(f"  validation_status     : {validation.get('status', 'unknown')}")
    print(f"  validation_device     : {validation.get('device_type', 'unknown')}")
    print(f"  dtype_used            : {validation.get('amp_dtype_used', 'unknown')}")
    print(f"  report_path           : {os.path.abspath(resolved_report_path)}")
    print("  training_quality_claim: none")
    for limitation in report.get("limitations", []):
        print(f"  caveat                : {limitation}")
    print("=" * 60)
    if not report["ok"]:
        sys.exit(1)
