"""Informational deployment and hardware-profile CLI commands."""

from __future__ import annotations

from typing import Any, Callable

from cli.output import print_banner
from config import available_deployment_tiers, build_deployment_report
from security.validator import ValidationError


def _default_emit_json(_payload: dict[str, Any]) -> None:
    raise RuntimeError("emit_json callback is required for JSON output")


def build_deployment_info_report() -> dict[str, Any]:
    reports = [
        build_deployment_report(tier_name)
        for tier_name in available_deployment_tiers()
    ]
    return {
        "schema": "deployment_tiers_v1",
        "reports": reports,
        "quality_claim": "none",
    }


def run_deployment_info(
    output_json: bool = False,
    emit_json: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    report = build_deployment_info_report()
    if output_json:
        (emit_json or _default_emit_json)(report)
        return

    print_banner("DEPLOYMENT TIERS")

    for item in report["reports"]:
        tier = item["tier"]
        estimates = item["estimates"]
        print(f"\n[{tier['name']}]")
        print(f"  target_vram_gb      : {tier['target_vram_gb']}")
        print(f"  model_profile       : {item['model_profile']}")
        print(f"  size_band           : {tier['intended_model_size_band']}")
        print(f"  active_context      : {tier['active_context_tokens']} tokens")
        print(
            "  weight_lower_bound : "
            f"{estimates['weight_memory_lower_bound_gb']} GB"
        )
        print(
            "  kv_lower_bound     : "
            f"{estimates['kv_cache_lower_bound_gb']} GB"
        )
        print("  runtime             : PyTorch reference path only")
        print("  external runtimes   : unverified")
        print(f"  quantization        : {tier['quantization_expectation']}")
        print(f"  offload             : {tier['offload_expectation']}")
        print(f"  rag                 : {tier['rag_expectation']}")

    print("\nCAVEAT: estimates are lower bounds only, not deployment proof.")
    print("CAVEAT: GGUF/GPTQ/AWQ/vLLM/llama.cpp are not implemented here.")
    print("=" * 60)


def build_hw_profile_report() -> dict[str, Any]:
    import torch
    import hardware_profiles
    from config import hardware_profile_cfg

    warnings = []
    profile = None
    profile_status = "loaded"
    try:
        profile = hardware_profiles.load_profile()
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        profile_status = "load_failed"
        warnings.append(str(exc))

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = []
    for index in range(device_count):
        device = {
            "index": index,
            "name": None,
            "total_memory_gb": None,
            "compute_capability": None,
        }
        try:
            props = torch.cuda.get_device_properties(index)
            device["name"] = str(getattr(props, "name", ""))
            total_memory = getattr(props, "total_memory", None)
            if total_memory is not None:
                device["total_memory_gb"] = round(
                    float(total_memory) / (1024 ** 3),
                    3,
                )
            capability = torch.cuda.get_device_capability(index)
            device["compute_capability"] = ".".join(
                str(part) for part in capability
            )
        except RuntimeError as exc:
            warnings.append(f"cuda_device_{index}_inspection_failed: {exc}")
        devices.append(device)

    return {
        "schema": "hardware_profile_v1",
        "command": "hw-profile",
        "profile_status": profile_status,
        "torch_available": True,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
        "selected_profile": profile.get("name") if profile else None,
        "active_profile_config": hardware_profile_cfg.active_profile,
        "profile": profile,
        "warnings": warnings,
        "training_started": False,
        "phase_b_started": False,
        "cloud_used": False,
        "model_quality_claim": "none",
        "quality_claim": "none",
    }


def run_hw_profile(
    output_json: bool = False,
    emit_json: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    if output_json:
        (emit_json or _default_emit_json)(build_hw_profile_report())
        return

    print_banner("STEP 0: Hardware Profile")
    import hardware_profiles
    hardware_profiles.show_profile()
