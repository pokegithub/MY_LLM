"""GPU-fit validation CLI command wrapper."""

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


def run_gpu_fit_validate(
    output_json: bool = False,
    emit_json: Callable[[dict[str, Any]], None] | None = None,
    write_json_report: Callable[[dict[str, Any], str], str] | None = None,
    report_path: str | None = None,
    default_report_path: str = "./run_artifacts/gpu_constrained_fit_report.json",
) -> None:
    """Report bounded 4GB GPU fit evidence without claiming training readiness."""
    import train

    report = train.run_gpu_constrained_fit_validation()
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

    print_banner("GPU CONSTRAINED-FIT VALIDATION")
    interpreter = report.get("interpreter", {})
    print(f"  interpreter        : {interpreter.get('executable', 'unknown')}")
    print(f"  repo_venv          : {interpreter.get('is_repo_local_venv', False)}")
    print(f"  cuda_available     : {report['cuda_available']}")
    print(f"  torch_version      : {report['torch_version']}")
    print(f"  torch_cuda_runtime : {report['torch_cuda_runtime']}")
    print(f"  gpu_name           : {report.get('gpu_name')}")
    print(f"  vram_gb            : {report.get('detected_total_vram_gb')}")
    print(f"  hardware_class     : {report['hardware_classification']}")
    print(f"  preferred_dtype    : {report['preferred_dtype']}")
    print("\n  Fit matrix:")
    for item in report.get("matrix", []):
        peak = item.get("peak_allocated_gb", "unknown")
        reserved = item.get("peak_reserved_gb", "unknown")
        print(
            "    [{fit:<23}] {name:<32} dtype={dtype:<8} "
            "seq={seq:<4} peak={peak}GB reserved={reserved}GB".format(
                fit=item.get("fit_classification", "unknown"),
                name=item.get("name", "unknown"),
                dtype=item.get("dtype_requested", "unknown"),
                seq=item.get("seq_len", "unknown"),
                peak=peak,
                reserved=reserved,
            )
        )
        if item.get("error"):
            print(f"      error: {str(item['error'])[:180]}")
    print(f"\n  smallest_fit       : {report.get('smallest_fitting_config')}")
    print(f"  strongest_fit      : {report.get('strongest_fitting_config')}")
    short = report.get("short_validation") or {}
    print(
        "  short_validation   : "
        f"{short.get('fit_classification', 'not_run')}"
    )
    if short:
        print(f"  short_steps        : {short.get('total_optimizer_steps', 'unknown')}")
        print(f"  short_peak_memory  : {short.get('memory_peak_allocated_bytes', 'unknown')}")
        print(f"  resume_success     : {short.get('resume_success', False)}")
    default_fit = report.get("default_training_fit") or {}
    print(f"  default_fit        : {default_fit.get('fit_classification', 'unknown')}")
    print(f"  pretrain_changed   : {report['pretraining_readiness_changed']}")
    print(f"  report_path        : {os.path.abspath(resolved_report_path)}")
    print("  quality_claim      : none")
    for limitation in report.get("limitations", []):
        print(f"  caveat             : {limitation}")
    print("=" * 60)
    if not report["ok"]:
        sys.exit(1)
