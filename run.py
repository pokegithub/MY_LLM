"""
run.py — Master orchestrator for the full SLM pipeline

Usage:
  python run.py agent-plan     Build a machine-readable agent plan
  python run.py agent-solve    Run the Phase 1 verified coding-agent solve path
  python run.py agent-verify   Run the Phase 1 verified coding-agent verification path
  python run.py tokenizer      Train the BPE tokenizer
  python run.py download       Download and cache training data
    python run.py download-safe  Download with network-safe defaults
        python run.py download-core  Download core-first bootstrap set
  python run.py download-status Show download manifest summary
  python run.py token-manifest Reconstruct token artifact manifest from local cache
  python run.py token-integrity Inspect token artifact integrity metadata
  python run.py deps           Check declared dependency environment
  python run.py hardware-validate Report current-machine hardware validation evidence
  python run.py gpu-fit-validate Report bounded 4GB GPU fit evidence
  python run.py data-governance Report source/license, dedup, and benchmark-risk evidence
  python run.py validate-real-path Run tiny real-token validation without pretraining
  python run.py validate-short-run Run bounded multi-step real-token validation
  python run.py train-preflight Check training readiness without starting training
  python run.py deployment-info Show truthful deployment tier metadata
  python run.py train          Pretrain the base model
  python run.py sft            Supervised fine-tuning
  python run.py dpo            Direct Preference Optimization
    python run.py distill        Distill student from teacher checkpoint
    python run.py quantize       Build quantized export artifacts
    python run.py hw-profile     Show active hardware profile
    python run.py eval-harness   Run evaluation with manifest harness
  python run.py improve        Self-improvement loop
  python run.py eval           Run evaluation suite
    python run.py benchmark-harness Run evaluation report; external scale comparison disabled/unverified
  python run.py audit          Run production-readiness audit checks
  python run.py full           Run the entire pipeline end-to-end
  python run.py status         Show pipeline status
"""

import os
import sys
import time
import json
import argparse
import importlib.util
import subprocess

from core.config_manager import ConfigManager
from core.dependency_checks import check_requirement_file
from config import apply_overrides, runtime_config_dict
from security.validator import ValidationError, get_allowed_data_roots, safe_load_json


OUTPUT_JSON = False
DEFAULT_TINY_REPORT_PATH = os.path.join(
    ".",
    "run_artifacts",
    "tiny_real_data_validation_report.json",
)
DEFAULT_HARDWARE_REPORT_PATH = os.path.join(
    ".",
    "run_artifacts",
    "hardware_readiness_report.json",
)
DEFAULT_GPU_FIT_REPORT_PATH = os.path.join(
    ".",
    "run_artifacts",
    "gpu_constrained_fit_report.json",
)
DEFAULT_DATA_GOVERNANCE_REPORT_PATH = os.path.join(
    ".",
    "run_artifacts",
    "data_governance_report.json",
)
DEFAULT_SHORT_VALIDATION_REPORT_PATH = os.path.join(
    ".",
    "run_artifacts",
    "short_real_data_validation_report.json",
)
DEFAULT_AGENT_REPORT_ROOT = os.path.join(
    ".",
    "run_artifacts",
    "agent",
)
REPORT_PATH = None


def _emit_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_json_report(payload, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def apply_runtime_config(
    config_paths=None,
    include_env: bool = True,
    strict: bool = False,
):
    """Load and apply runtime config layers before command dispatch."""
    manager = ConfigManager()
    snapshot = manager.load(config_paths=config_paths or [], include_env=include_env)
    if not snapshot.sources:
        return None

    result = apply_overrides(snapshot.data, strict=strict)

    os.makedirs("./run_artifacts", exist_ok=True)
    stamp = int(time.time())
    snapshot_path = os.path.join(
        "./run_artifacts", f"runtime_config_snapshot_{stamp}.json"
    )
    state_path = os.path.join(
        "./run_artifacts", f"runtime_config_effective_{stamp}.json"
    )

    ConfigManager.save_json(snapshot, snapshot_path)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config_dict(), f, indent=2)

    print("\n" + "=" * 60)
    print("RUNTIME CONFIG")
    print("=" * 60)
    print(f"  sources      : {', '.join(snapshot.sources)}")
    print(f"  config_hash  : {snapshot.config_hash}")
    print(f"  applied_keys : {result['applied']}")
    if result["unknown_sections"]:
        print(
            "  unknown sections: "
            + ", ".join(result["unknown_sections"])
        )
    if result["unknown_fields"]:
        print(
            "  unknown fields  : "
            + ", ".join(result["unknown_fields"])
        )
    print(f"  snapshot     : {os.path.abspath(snapshot_path)}")
    print(f"  effective    : {os.path.abspath(state_path)}")
    print("=" * 60)

    return snapshot


def check_dependencies(command: str):
    """Check required packages for the selected command."""
    required_by_cmd = {
        "agent-plan": {},
        "agent-solve": {},
        "agent-verify": {},
        "tokenizer": {"tokenizers": "tokenizers", "datasets": "datasets"},
        "download": {"numpy": "numpy", "datasets": "datasets"},
        "download-safe": {"numpy": "numpy", "datasets": "datasets"},
        "download-core": {"numpy": "numpy", "datasets": "datasets"},
        "token-manifest": {"numpy": "numpy", "tokenizers": "tokenizers"},
        "token-integrity": {"numpy": "numpy"},
        "deps": {},
        "hardware-validate": {"torch": "torch"},
        "gpu-fit-validate": {"torch": "torch", "numpy": "numpy", "tokenizers": "tokenizers"},
        "data-governance": {"numpy": "numpy"},
        "validate-real-path": {"torch": "torch", "numpy": "numpy", "tokenizers": "tokenizers", "datasets": "datasets"},
        "validate-short-run": {"torch": "torch", "numpy": "numpy", "tokenizers": "tokenizers"},
        "train-preflight": {"torch": "torch", "numpy": "numpy"},
        "deployment-info": {},
        "train": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "sft": {"torch": "torch", "numpy": "numpy"},
        "dpo": {"torch": "torch", "numpy": "numpy"},
        "distill": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "quantize": {"torch": "torch", "numpy": "numpy"},
        "hw-profile": {"torch": "torch"},
        "eval-harness": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "improve": {"torch": "torch", "numpy": "numpy"},
        "eval": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "benchmark-harness": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "full": {"torch": "torch", "numpy": "numpy", "datasets": "datasets", "tokenizers": "tokenizers"},
        "audit": {"torch": "torch", "numpy": "numpy", "tokenizers": "tokenizers"},
        "download-status": {},
        "status": {},
    }
    required = required_by_cmd.get(command, {})
    optional = {
        "regex": "regex",
    }
    missing = []

    def _is_available(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    for name, pkg in required.items():
        if not _is_available(pkg):
            missing.append(name)

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    for name, pkg in optional.items():
        if not _is_available(pkg):
            if not OUTPUT_JSON:
                print(f"  TIP: pip install {name}  (for better tokenization)")


STATUS_TARGETS = [
    ("Tokenizer", "./tokenizer_data/encoder.json"),
    ("Data cache", "./data_cache/tokens"),
    ("Val cache", "./data_cache/val_tokens"),
    ("Pretrain ckpt", "./checkpoints"),
    ("SFT ckpt", "./sft_checkpoints"),
    ("DPO ckpt", "./dpo_checkpoints"),
    ("Improved ckpt", "./improved_checkpoints"),
    ("Eval results", "./eval_results/eval_report.json"),
]


def collect_status_checks():
    """Collect filesystem-backed status rows without printing."""
    rows = []
    for name, path in STATUS_TARGETS:
        row = {
            "name": name,
            "path": path,
            "exists": os.path.exists(path),
            "is_dir": os.path.isdir(path),
            "file_count": 0,
            "size_mb": 0.0,
        }
        if row["exists"]:
            if row["is_dir"]:
                files = os.listdir(path)
                row["file_count"] = len(files)
                row["size_mb"] = sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in files
                    if os.path.isfile(os.path.join(path, f))
                ) / 1e6
            else:
                row["size_mb"] = os.path.getsize(path) / 1e6
        rows.append(row)
    return rows


def collect_dependency_checks(requirements_path: str = "./requirement.txt"):
    """Collect declared dependency drift for status/preflight visibility."""
    return check_requirement_file(requirements_path)


def _print_dependency_summary(report):
    print("\nDEPENDENCY ENVIRONMENT")
    print(f"  requirements: {os.path.abspath(report['path'])}")
    if not report["exists"]:
        print("  [UNVERIFIED] requirement file is missing")
        return
    summary = report["summary"]
    print(
        "  summary     : "
        f"pass={summary['pass']} fail={summary['fail']} "
        f"unverified={summary['unverified']}"
    )
    for item in report["items"]:
        status = str(item["status"]).upper()
        installed = item["installed"] if item["installed"] is not None else "missing"
        print(
            f"  [{status:<10}] {item['package']:<12} "
            f"installed={installed} required={item['required']}"
        )


def _next_status_action(dependency_ok: bool):
    if not os.path.exists("./tokenizer_data/encoder.json"):
        return {
            "command": "python run.py tokenizer",
            "ready_for_training": False,
            "reason": "tokenizer artifacts are missing",
        }
    if not os.path.exists("./data_cache/tokens"):
        return {
            "command": "python run.py download",
            "ready_for_training": False,
            "reason": "token cache is missing",
        }
    if not os.path.exists("./data_cache/token_artifacts_manifest.json"):
        return {
            "command": "python run.py token-manifest",
            "ready_for_training": False,
            "reason": "token artifact manifest is missing",
        }
    if not dependency_ok:
        return {
            "command": "fix dependency drift, then run: python run.py train-preflight",
            "ready_for_training": False,
            "reason": "declared dependency environment is not satisfied",
        }
    if not os.path.exists("./checkpoints"):
        return {
            "command": "python run.py train-preflight",
            "ready_for_training": False,
            "reason": "preflight must pass before training; checkpoint directory is missing",
        }
    if not os.path.exists("./sft_checkpoints"):
        return {
            "command": "python run.py sft",
            "ready_for_training": False,
            "reason": "base checkpoint exists; SFT checkpoint is missing",
        }
    if not os.path.exists("./dpo_checkpoints"):
        return {
            "command": "python run.py dpo",
            "ready_for_training": False,
            "reason": "SFT checkpoint exists; DPO checkpoint is missing",
        }
    return {
        "command": "python run.py eval",
        "ready_for_training": False,
        "reason": "training artifacts exist; evaluation is next",
    }


def build_status_report():
    dependency_report = collect_dependency_checks()
    return {
        "schema": "pipeline_status_v2",
        "status_checks": collect_status_checks(),
        "dependency_environment": dependency_report,
        "next_action": _next_status_action(dependency_report["ok"]),
        "training_quality_claim": "none",
    }


def show_status():
    """Show current pipeline status."""
    report = build_status_report()
    if OUTPUT_JSON:
        _emit_json(report)
        return

    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    for row in collect_status_checks():
        name = row["name"]
        if row["exists"]:
            if row["is_dir"]:
                print(
                    f"  [OK]      {name:<20} "
                    f"{row['file_count']} files, {row['size_mb']:.1f}MB"
                )
            else:
                print(f"  [OK]      {name:<20} {row['size_mb']:.1f}MB")
        else:
            print(f"  [MISSING] {name:<20} not found")

    print("=" * 60)
    dependency_report = report["dependency_environment"]
    _print_dependency_summary(dependency_report)

    next_action = report["next_action"]
    print(f"\n  NEXT: {next_action['command']}")
    if next_action["command"] == "python run.py token-manifest":
        print("  Training remains NOT READY until train-preflight passes.")
    elif not dependency_report["ok"]:
        print("  Do not start training from this environment yet.")
    elif next_action["command"] == "python run.py train-preflight":
        print("  Only run python run.py train after preflight passes.")


def run_deps():
    """Show declared dependency satisfaction."""
    report = collect_dependency_checks()
    if OUTPUT_JSON:
        _emit_json(report)
        if not report["ok"]:
            sys.exit(1)
        return

    _print_dependency_summary(report)
    if not report["ok"]:
        print("\nDEPENDENCY CHECK FAILED")
        sys.exit(1)
    print("\nDEPENDENCY CHECK PASSED")


def _load_task_text(task: str | None, task_file: str | None) -> str:
    if task and task_file:
        raise ValueError("provide either --task or --task-file, not both")
    if task_file:
        with open(task_file, "r", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = task or ""
    if not text.strip():
        raise ValueError("agent command requires non-empty --task or --task-file")
    return text


def _build_agent_request(args):
    from agent.types import TaskRequest

    task_text = _load_task_text(args.task, args.task_file)
    return TaskRequest(
        task_text=task_text,
        file_hints=tuple(args.file_hint or ()),
        checks=tuple(args.check or ()),
        workspace_root=os.getcwd(),
        task_file=args.task_file,
    )


def _print_agent_report(title: str, payload: dict):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"  run_id         : {payload['run_id']}")
    print(f"  route          : {payload['route']['route']}")
    print(f"  status         : {payload['status']}")
    print(f"  backend        : {payload['backend_status'].get('kind', 'unknown')}")
    print(f"  report_path    : {os.path.abspath(payload['report_path'])}")
    if payload.get("blocked_reason"):
        print(f"  blocked_reason : {payload['blocked_reason']}")
    if payload.get("degraded_mode"):
        print("  degraded_mode  : true")
    artifact_warnings = payload.get("artifact_warnings") or []
    if artifact_warnings:
        print(f"  warnings       : {len(artifact_warnings)} artifact warning(s)")
    print(f"  quality_claim  : {payload['quality_claim']}")
    print("=" * 60)


def run_agent_plan(args):
    from agent.orchestrator import plan_task

    payload = plan_task(_build_agent_request(args))
    if OUTPUT_JSON:
        _emit_json(payload)
        return
    _print_agent_report("AGENT PLAN", payload)


def run_agent_solve(args):
    from agent.orchestrator import solve_task

    payload = solve_task(
        _build_agent_request(args),
        backend_script_path=args.backend_script,
    )
    if OUTPUT_JSON:
        _emit_json(payload)
        return
    _print_agent_report("AGENT SOLVE", payload)


def run_agent_verify(args):
    from agent.orchestrator import verify_task

    payload = verify_task(_build_agent_request(args))
    if OUTPUT_JSON:
        _emit_json(payload)
        return
    _print_agent_report("AGENT VERIFY", payload)


def run_hardware_validate():
    """Report current-machine hardware validation evidence."""
    import train

    report = train.run_hardware_readiness_validation()
    report_path = _write_json_report(
        report,
        REPORT_PATH or DEFAULT_HARDWARE_REPORT_PATH,
    )
    report["report_path"] = report_path
    if OUTPUT_JSON:
        _emit_json(report)
        if not report["ok"]:
            sys.exit(1)
        return

    print("\n" + "=" * 60)
    print("HARDWARE VALIDATION")
    print("=" * 60)
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
    print(f"  report_path           : {os.path.abspath(report_path)}")
    print("  training_quality_claim: none")
    for limitation in report.get("limitations", []):
        print(f"  caveat                : {limitation}")
    print("=" * 60)
    if not report["ok"]:
        sys.exit(1)


def run_gpu_fit_validate():
    """Report bounded 4GB GPU fit evidence without claiming training readiness."""
    import train

    report = train.run_gpu_constrained_fit_validation()
    report_path = _write_json_report(
        report,
        REPORT_PATH or DEFAULT_GPU_FIT_REPORT_PATH,
    )
    report["report_path"] = report_path
    if OUTPUT_JSON:
        _emit_json(report)
        if not report["ok"]:
            sys.exit(1)
        return

    print("\n" + "=" * 60)
    print("GPU CONSTRAINED-FIT VALIDATION")
    print("=" * 60)
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
    print(f"  report_path        : {os.path.abspath(report_path)}")
    print("  quality_claim      : none")
    for limitation in report.get("limitations", []):
        print(f"  caveat             : {limitation}")
    print("=" * 60)
    if not report["ok"]:
        sys.exit(1)


def run_data_governance():
    """Report source/license, exact-dedup, and benchmark-risk evidence."""
    import download_data

    report = download_data.build_data_governance_report()
    report_path = _write_json_report(
        report,
        REPORT_PATH or DEFAULT_DATA_GOVERNANCE_REPORT_PATH,
    )
    report["report_path"] = report_path
    if OUTPUT_JSON:
        _emit_json(report)
        if not report["ok"]:
            sys.exit(1)
        return

    source_manifest = report["source_manifest"]
    dedup = report["exact_dedup"]
    benchmark_risk = report["benchmark_source_risk"]
    summary = report.get("summary", {})
    print("\n" + "=" * 60)
    print("DATA GOVERNANCE EVIDENCE")
    print("=" * 60)
    print(f"  sources               : {source_manifest['source_count']}")
    print(f"  active_sources        : {source_manifest['active_source_count']}")
    print(f"  known_licenses        : {source_manifest['known_license_count']}")
    print(f"  documented_licenses   : {source_manifest.get('documented_license_count', source_manifest['known_license_count'])}")
    print(f"  unknown_licenses      : {source_manifest['unknown_license_count']}")
    manual = source_manifest.get("manual_metadata", {})
    print(f"  manual_metadata       : {manual.get('matched_source_count', 0)}/{source_manifest['source_count']} sources")
    print(f"  repo_policy_counts    : {source_manifest.get('repo_policy_status_counts', {})}")
    print(f"  blocker_counts        : {source_manifest.get('training_blocker_level_counts', {})}")
    print(f"  active_hard_blockers  : {source_manifest.get('active_hard_blocker_count', 'unknown')}")
    print(f"  allowed_by_policy     : {len(source_manifest.get('allowed_by_repo_policy_sources', []))}")
    print(f"  policy_blocks_training: {source_manifest.get('serious_training_blocked_by_policy', True)}")
    print(f"  license_counts        : {source_manifest.get('declared_license_counts', {})}")
    print(f"  governance_counts     : {source_manifest.get('governance_classification_counts', {})}")
    print(f"  dedup_scope           : {dedup.get('scope', 'unknown')}")
    print(f"  chunks_available_est  : {dedup.get('chunks_available_estimate', 'unknown')}")
    print(f"  exact_chunks_checked  : {dedup['chunks_inspected']}")
    print(f"  exact_duplicate_chunks: {dedup['duplicate_chunks']}")
    print(f"  duplicate_ratio       : {dedup['duplicate_ratio']}")
    tv = dedup.get("train_val_exact_overlap", {})
    print(f"  train_val_overlap    : {tv.get('overlap_chunks', 'unknown')}")
    source_risk = benchmark_risk.get("source_level_risk", {})
    content_overlap = benchmark_risk.get("content_level_overlap", {})
    print(f"  benchmark_source_hits : {source_risk.get('risk_artifact_count', benchmark_risk['source_artifact_risk_count'])}")
    print(f"  source_risk_claim     : {source_risk.get('claim', 'source_level_risk_only')}")
    print(f"  content_overlap       : {content_overlap.get('status', benchmark_risk['content_overlap_status'])}")
    print(f"  report_path           : {os.path.abspath(report_path)}")
    print("  legal_clearance_claim : none")
    print("  contamination_claim   : none")
    print(f"  training_ready_change : {summary.get('training_readiness_changed', 'no')}")
    print("=" * 60)
    if not report["ok"]:
        sys.exit(1)


def run_tokenizer():
    print("\n" + "=" * 60)
    print("STEP 1: Training Tokenizer")
    print("=" * 60)
    t0 = time.time()

    import train_tokenizer
    train_tokenizer.main()

    elapsed = time.time() - t0
    print(f"\nTokenizer training complete: {elapsed / 60:.1f} min")


def run_download():
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Data")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer not found. Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    import download_data
    download_data.main()

    elapsed = time.time() - t0
    print(f"\nData download complete: {elapsed / 60:.1f} min")


def run_download_safe():
    """Run downloader with safer defaults for flaky networks."""
    os.environ.setdefault("DOWNLOAD_AUTO_PROCEED", "1")
    os.environ.setdefault("DOWNLOAD_NETWORK_SAFE", "1")
    os.environ.setdefault("DOWNLOAD_MAX_SOURCE_FAILURES", "3")
    print("\nUsing network-safe download defaults:")
    print("  DOWNLOAD_AUTO_PROCEED=1")
    print("  DOWNLOAD_NETWORK_SAFE=1")
    print("  DOWNLOAD_MAX_SOURCE_FAILURES=3")
    run_download()


def run_download_core():
    """Run downloader in core-first mode to bootstrap usable data quickly."""
    os.environ.setdefault("DOWNLOAD_AUTO_PROCEED", "1")
    os.environ.setdefault("DOWNLOAD_NETWORK_SAFE", "1")
    os.environ.setdefault("DOWNLOAD_CORE_FIRST", "1")
    os.environ.setdefault("DOWNLOAD_SOURCE_LIMIT", "10")
    os.environ.setdefault("DOWNLOAD_MAX_SOURCE_FAILURES", "2")
    print("\nUsing core-first download defaults:")
    print("  DOWNLOAD_AUTO_PROCEED=1")
    print("  DOWNLOAD_NETWORK_SAFE=1")
    print("  DOWNLOAD_CORE_FIRST=1")
    print("  DOWNLOAD_SOURCE_LIMIT=10")
    print("  DOWNLOAD_MAX_SOURCE_FAILURES=2")
    run_download()


def run_download_status():
    """Print summary from download manifest for operational visibility."""
    manifest_path = os.path.join(".", "data_cache", "download_manifest.json")
    summary = {
        "total_runs": 0,
        "recent": [],
        "fail_counts": {},
    }
    if os.path.exists(manifest_path):
        try:
            manifest = safe_load_json(
                manifest_path,
                max_bytes=8_000_000,
                allowed_roots=get_allowed_data_roots(),
            )
            runs = manifest.get("runs", []) if isinstance(manifest, dict) else []
            recent = runs[-5:]
            fail_counts = {}
            for r in recent:
                for src in r.get("failed_sources", []):
                    fail_counts[src] = fail_counts.get(src, 0) + 1
            summary = {
                "total_runs": len(runs),
                "recent": recent,
                "fail_counts": dict(
                    sorted(fail_counts.items(), key=lambda kv: kv[1], reverse=True)
                ),
            }
        except (ValidationError, OSError, ValueError) as e:
            print(f"\nFailed reading manifest: {str(e)[:120]}")
            return

    print("\n" + "=" * 60)
    print("DOWNLOAD STATUS")
    print("=" * 60)
    print(f"  Manifest: {os.path.abspath(manifest_path)}")
    print(f"  Total runs tracked: {summary['total_runs']}")

    recent = summary.get("recent", [])
    if not recent:
        print("  No download runs recorded yet.")
    else:
        print("  Recent runs:")
        for r in recent:
            run_id = r.get("run_id", "?")
            new_tok = int(r.get("total_new_tokens", 0))
            cache_gb = r.get("cache_gb", "?")
            failed = len(r.get("failed_sources", []))
            print(
                f"    run_id={run_id} | new_tokens={new_tok:,} | "
                f"cache={cache_gb}GB | failed_sources={failed}"
            )

    fail_counts = summary.get("fail_counts", {})
    if fail_counts:
        print("  Top failing sources:")
        for src, cnt in list(fail_counts.items())[:10]:
            print(f"    {src}: {cnt}")
    print("=" * 60)


def run_token_manifest():
    """Reconstruct token artifact manifest from existing local cache."""
    import download_data

    try:
        report = download_data.reconstruct_token_artifact_manifest()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: failed to reconstruct token artifact manifest: {exc}")
        sys.exit(1)

    if OUTPUT_JSON:
        _emit_json({
            "schema": "token_artifact_manifest_reconstruction_v1",
            **report,
            "quality_claim": "none",
        })
        if report["artifact_count"] == 0:
            sys.exit(1)
        return

    print("\n" + "=" * 60)
    print("TOKEN ARTIFACT MANIFEST")
    print("=" * 60)
    print(f"  manifest_path       : {os.path.abspath(report['manifest_path'])}")
    print(f"  artifact_count      : {report['artifact_count']}")
    print(f"  unknown_source_count: {report['unknown_source_count']}")
    print("  provenance          : reconstructed_from_local_cache")
    print("  quality_claim       : none")
    if report["artifact_count"] == 0:
        print("ERROR: no local token artifacts found")
        sys.exit(1)
    print("=" * 60)


def run_token_integrity():
    """Inspect token artifacts from the local token manifest."""
    import download_data

    report = download_data.validate_token_artifact_manifest()
    if OUTPUT_JSON:
        _emit_json(report)
        if not report["ok"]:
            sys.exit(1)
        return

    print("\n" + "=" * 60)
    print("TOKEN ARTIFACT INTEGRITY")
    print("=" * 60)
    print(f"  manifest_path          : {os.path.abspath(report['manifest_path'])}")
    print(f"  schema                 : {report['schema']}")
    print(f"  artifact_count         : {report['artifact_count']}")
    print(f"  existing_artifact_count: {report['existing_artifact_count']}")
    print(f"  failed_artifact_count  : {report['failed_artifact_count']}")
    print(f"  missing train/val pairs: {len(report['missing_train_val_pairs'])}")
    print(f"  hash_scope             : {report['hash_scope']}")
    print("  quality_claim          : none")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print("=" * 60)
    if not report["ok"]:
        sys.exit(1)


def run_train():
    print("\n" + "=" * 60)
    print("STEP 3: Pretraining")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer not found. Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    import train
    try:
        train.assert_training_preflight_ready(train.run_training_preflight())
    except train.TrainingPreflightError as exc:
        print(f"ERROR: training preflight is not ready: {exc}")
        print("Run: python run.py train-preflight")
        sys.exit(1)
    train.train()

    elapsed = time.time() - t0
    print(f"\nPretraining complete: {elapsed / 60:.1f} min")


def run_train_preflight():
    import train

    report = train.run_training_preflight()
    if OUTPUT_JSON:
        _emit_json(report)
        if not report["ready_for_training"]:
            sys.exit(1)
        return

    print("\n" + "=" * 60)
    print("TRAINING PREFLIGHT")
    print("=" * 60)

    for item in report["checks"]:
        status = str(item.get("status", "unknown")).upper()
        print(f"  [{status:<10}] {item['name']}: {item['detail']}")

    summary = report["summary"]
    print("=" * 60)
    print(
        "  failures={failures} warnings={warnings} unverified={unverified}".format(
            **summary
        )
    )
    if report["ready_for_training"]:
        print("  TRAINING PREFLIGHT PASSED")
        return

    print("  TRAINING PREFLIGHT NOT READY")
    print("  This does not start training and does not prove training quality.")
    sys.exit(1)


def run_validate_real_path():
    """Run tiny real-token validation without starting pretraining."""
    import train

    try:
        report = train.run_tiny_real_data_validation(
            checkpoint_dir="./run_artifacts/tiny_real_data_validation",
            seq_len=16,
            deterministic=True,
            data_loader_smoke=True,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        payload = {
            "schema": "tiny_real_data_validation_v1",
            "ok": False,
            "error": str(exc),
            "quality_claim": "none",
        }
        if OUTPUT_JSON:
            _emit_json(payload)
        else:
            print(f"ERROR: tiny real-data validation failed: {exc}")
        sys.exit(1)

    report["report_path"] = _write_json_report(
        report,
        REPORT_PATH or DEFAULT_TINY_REPORT_PATH,
    )
    if OUTPUT_JSON:
        _emit_json(report)
        return

    print("\n" + "=" * 60)
    print("TINY REAL-DATA VALIDATION")
    print("=" * 60)
    print(f"  artifact_path     : {report['artifact_path']}")
    print(f"  tokens_read       : {report['tokens_read']}")
    print(f"  checkpoint_path   : {report['checkpoint_path']}")
    print(f"  resumed_step      : {report['resumed_step']}")
    print(f"  loss              : {report['loss']:.6f}")
    print(f"  data_loader_smoke : {report['data_loader_smoke']['status']}")
    print(f"  eval gate         : {report['eval_missing_checkpoint_gate']}")
    print(f"  report_path       : {os.path.abspath(report['report_path'])}")
    print("  quality_claim     : none")
    print("=" * 60)


def run_validate_short_run():
    """Run bounded multi-step real-token validation without pretraining."""
    import train

    try:
        report = train.run_short_real_data_validation(
            checkpoint_dir="./run_artifacts/short_real_data_validation",
            seq_len=16,
            pre_resume_steps=3,
            post_resume_steps=2,
            deterministic=True,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        payload = {
            "schema": "short_real_data_validation_v1",
            "ok": False,
            "error": str(exc),
            "quality_claim": "none",
            "run_classification": "bounded_validation_not_pretraining",
        }
        if OUTPUT_JSON:
            _emit_json(payload)
        else:
            print(f"ERROR: short real-data validation failed: {exc}")
        sys.exit(1)

    report_path = _write_json_report(
        report,
        REPORT_PATH or DEFAULT_SHORT_VALIDATION_REPORT_PATH,
    )
    report["report_path"] = report_path
    if OUTPUT_JSON:
        _emit_json(report)
        return

    print("\n" + "=" * 60)
    print("SHORT REAL-DATA VALIDATION")
    print("=" * 60)
    print(f"  classification     : {report['run_classification']}")
    print(f"  artifact_path      : {report['artifact_path']}")
    print(f"  total_steps        : {report['total_optimizer_steps']}")
    print(f"  device             : {report['device']}")
    print(f"  dtype_used         : {report['amp_dtype_used']}")
    print(f"  resumed_step       : {report['resumed_step']}")
    print(f"  resume_success     : {report['resume_success']}")
    print(f"  final_checkpoint   : {report['final_checkpoint']}")
    print(f"  eval gate          : {report['eval_gate_on_produced_checkpoint']}")
    print(f"  missing eval gate  : {report['eval_missing_checkpoint_gate']}")
    print(f"  report_path        : {os.path.abspath(report_path)}")
    print("  quality_claim      : none")
    print("=" * 60)


def run_deployment_info():
    from config import available_deployment_tiers, build_deployment_report

    reports = [
        build_deployment_report(tier_name)
        for tier_name in available_deployment_tiers()
    ]
    if OUTPUT_JSON:
        _emit_json({
            "schema": "deployment_tiers_v1",
            "reports": reports,
            "quality_claim": "none",
        })
        return

    print("\n" + "=" * 60)
    print("DEPLOYMENT TIERS")
    print("=" * 60)

    for report in reports:
        tier = report["tier"]
        estimates = report["estimates"]
        print(f"\n[{tier['name']}]")
        print(f"  target_vram_gb      : {tier['target_vram_gb']}")
        print(f"  model_profile       : {report['model_profile']}")
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


def run_sft():
    print("\n" + "=" * 60)
    print("STEP 4: Supervised Fine-Tuning")
    print("=" * 60)

    if not os.path.exists("./sft_data"):
        print("SFT data directory not found: ./sft_data/")
        print("\nExpected format (JSONL):")
        print('  {"messages": [')
        print('    {"role": "system", "content": "You are helpful."},')
        print('    {"role": "user", "content": "Hello!"},')
        print('    {"role": "assistant", "content": "Hi there!"}')
        print("  ]}")
        print("\nCreate ./sft_data/ with your JSONL files and re-run.")

        os.makedirs("./sft_data", exist_ok=True)
        sample = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ]
            },
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful coding assistant.",
                    },
                    {
                        "role": "user",
                        "content": "Write hello world in Python.",
                    },
                    {
                        "role": "assistant",
                        "content": 'print("Hello, World!")',
                    },
                ]
            },
        ]
        with open("./sft_data/sample.jsonl", "w", encoding="utf-8") as f:
            for s in sample:
                f.write(json.dumps(s) + "\n")
        print("\nCreated sample: ./sft_data/sample.jsonl")
        return

    t0 = time.time()
    import sft_trainer
    sft_trainer.sft_train()

    elapsed = time.time() - t0
    print(f"\nSFT complete: {elapsed / 60:.1f} min")


def run_dpo():
    print("\n" + "=" * 60)
    print("STEP 5: Direct Preference Optimization")
    print("=" * 60)

    if not os.path.exists("./dpo_data"):
        print("DPO data directory not found: ./dpo_data/")
        print("\nExpected format (JSONL):")
        print('  {"prompt": "What is AI?",')
        print('   "chosen": "AI is artificial intelligence...",')
        print('   "rejected": "AI is magic..."}')
        print("\nCreate ./dpo_data/ with your JSONL files and re-run.")

        os.makedirs("./dpo_data", exist_ok=True)
        sample = [
            {
                "prompt": "Explain gravity simply.",
                "chosen": "Gravity is the force that pulls objects "
                          "toward each other. The more massive an "
                          "object, the stronger its gravitational pull.",
                "rejected": "Gravity is just what happens. Things fall "
                            "down because they want to.",
            },
        ]
        with open("./dpo_data/sample.jsonl", "w", encoding="utf-8") as f:
            for s in sample:
                f.write(json.dumps(s) + "\n")
        print("\nCreated sample: ./dpo_data/sample.jsonl")
        return

    t0 = time.time()
    import alignment
    alignment.dpo_train()

    elapsed = time.time() - t0
    print(f"\nDPO complete: {elapsed / 60:.1f} min")


def run_distill():
    print("\n" + "=" * 60)
    print("STEP 5.5: Distillation")
    print("=" * 60)
    t0 = time.time()

    import distillation_trainer
    distillation_trainer.distill_train()

    elapsed = time.time() - t0
    print(f"\nDistillation complete: {elapsed / 60:.1f} min")


def run_quantize():
    print("\n" + "=" * 60)
    print("STEP 6.5: Quantization")
    print("=" * 60)
    t0 = time.time()

    import quant_utils
    quant_utils.run_quantization()

    elapsed = time.time() - t0
    print(f"\nQuantization complete: {elapsed / 60:.1f} min")


def run_hw_profile():
    print("\n" + "=" * 60)
    print("STEP 0: Hardware Profile")
    print("=" * 60)
    import hardware_profiles
    hardware_profiles.show_profile()


def run_eval_harness():
    print("\n" + "=" * 60)
    print("STEP 7.5: Evaluation Harness")
    print("=" * 60)
    t0 = time.time()

    from eval_harness.runner import run
    run()

    elapsed = time.time() - t0
    print(f"\nEvaluation harness complete: {elapsed / 60:.1f} min")


def run_improve():
    print("\n" + "=" * 60)
    print("STEP 6: Self-Improvement Loop")
    print("=" * 60)

    t0 = time.time()
    import infinite_improver
    infinite_improver.improve()

    elapsed = time.time() - t0
    print(f"\nSelf-improvement complete: {elapsed / 60:.1f} min")


def run_eval():
    print("\n" + "=" * 60)
    print("STEP 7: Evaluation")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer artifacts not found.")
        print("Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    import eval_suite
    eval_suite.run_evaluation()

    elapsed = time.time() - t0
    print(f"\nEvaluation complete: {elapsed / 60:.1f} min")


def run_benchmark_harness():
    print("\n" + "=" * 60)
    print("STEP 7.2: Benchmark Harness")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer artifacts not found.")
        print("Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    from eval.benchmark_harness import run_benchmark_harness as _run
    _run()

    elapsed = time.time() - t0
    print(f"\nBenchmark harness complete: {elapsed / 60:.1f} min")


def audit_status_command_health():
    """Verify status is truthful enough to run cleanly in a subprocess."""
    cmd = [sys.executable, "run.py", "status"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"status command failed: {detail[:300]}")
    combined = result.stdout + result.stderr
    if "Traceback" in combined:
        raise RuntimeError("status command emitted a traceback")
    if "\u2713" in combined or "\u2717" in combined:
        raise RuntimeError("status output contains non-ASCII status glyphs")
    if "PIPELINE STATUS" not in result.stdout:
        raise RuntimeError("status output did not include the status header")
    if "NEXT: python run.py train\n" in result.stdout:
        raise RuntimeError("status output recommends training without preflight")


def audit_serving_contract():
    """Fail if the serving path still behaves like an echo scaffold."""
    from serving.server import handle_request

    result = handle_request("pass1 serving audit prompt")
    if not isinstance(result, dict):
        raise RuntimeError("serving handler did not return a dict")
    if result.get("ok") is True:
        response = str(result.get("response", ""))
        if response.startswith("Echo:"):
            raise RuntimeError("serving handler still returns echo responses")
        raise RuntimeError("serving returned success without a configured model backend")
    error = str(result.get("error", ""))
    if "model backend not loaded" not in error.lower():
        raise RuntimeError(
            "serving failure is not explicit about the missing model backend"
        )


def audit_quantization_report_contract():
    """Verify quantization reports cannot encode partial export as success."""
    from quant_utils import validate_quant_report_payload

    valid_failure = {
        "success": False,
        "quant_mode": "dynamic-int8",
        "errors": ["torchscript export failed"],
    }
    validate_quant_report_payload(valid_failure)

    invalid_success = {
        "success": True,
        "quant_mode": "dynamic-int8",
        "errors": ["torchscript export failed"],
    }
    try:
        validate_quant_report_payload(invalid_success)
    except ValueError:
        return
    raise RuntimeError("quantization report accepted success with errors")


def run_audit():
    """Run narrow Pass 1 truthfulness checks."""
    print("\n" + "=" * 60)
    print("PIPELINE AUDIT")
    print("=" * 60)

    checks = [
        (
            "Python compile check",
            [
                sys.executable,
                "-m",
                "compileall",
                "tokenizer.py",
                "alignment.py",
                "sft_trainer.py",
                "eval_suite.py",
                "infinite_improver.py",
                "model.py",
                "run.py",
            ],
        ),
        (
            "Tokenizer smoke test",
            [
                sys.executable,
                "-c",
                (
                    "from tokenizer import BPETokenizer; "
                    "t=BPETokenizer(); "
                    "t.encoder={t.PAD:0,t.BOS:1,t.EOS:2,t.UNK:3,t.USR:4,t.AST:5}; "
                    "t.decoder={v:k for k,v in t.encoder.items()}; "
                    "t.vocab_size_=len(t.encoder); "
                    "t._special_set=set(t.SPECIAL); t._compile_special_pattern(); "
                    "ids=t.encode('<|user|>x<|assistant|>y', add_bos=True); "
                    "print('OK', len(ids))"
                ),
            ],
        ),
    ]

    for name, cmd in checks:
        print(f"  Running: {name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            print(f"\nAUDIT FAILED: {name}")
            sys.exit(1)

    extra_checks = [
        ("Status command health", audit_status_command_health),
        ("Fake serving detection", audit_serving_contract),
        ("Quantization report integrity", audit_quantization_report_contract),
    ]
    for name, check in extra_checks:
        print(f"  Running: {name}")
        try:
            check()
        except RuntimeError as exc:
            print(f"\nAUDIT FAILED: {name}: {exc}")
            sys.exit(1)

    print("\nAUDIT PASSED")


def run_full():
    """Run the full pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)
    t_total = time.time()

    stages = [
        ("tokenizer", run_tokenizer),
        ("download", run_download),
        ("train", run_train),
        ("sft", run_sft),
        ("dpo", run_dpo),
        ("improve", run_improve),
        ("eval", run_eval),
    ]

    for name, fn in stages:
        try:
            fn()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            print("Pipeline stopped. Fix the issue and re-run.")
            sys.exit(1)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"FULL PIPELINE COMPLETE: {elapsed / 3600:.1f} hours")
    print(f"{'=' * 60}")


def main():
    global OUTPUT_JSON, REPORT_PATH

    parser = argparse.ArgumentParser(
        description="SLM Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  agent-plan   Build a machine-readable agent plan for one task
  agent-solve  Run the Phase 1 agent solve path
  agent-verify Run the Phase 1 agent verification path
  tokenizer    Train the BPE tokenizer
  download     Download and cache training data
    download-safe Download with network-safe defaults
        download-core Download core-first bootstrap set
    download-status Show download manifest summary
  token-manifest Reconstruct token artifact manifest from local cache
  token-integrity Inspect token artifact integrity metadata
  deps          Check declared dependency environment
  hardware-validate Report current-machine hardware validation evidence
  gpu-fit-validate Report bounded 4GB GPU fit evidence
  data-governance Report source/license, dedup, and benchmark-risk evidence
  validate-real-path Run tiny real-token validation without pretraining
  validate-short-run Run bounded multi-step real-token validation
  train-preflight Check training readiness without starting training
  deployment-info Show truthful deployment tier metadata
  train        Pretrain the base model
  sft          Supervised fine-tuning
  dpo          Direct Preference Optimization
    distill      Distill student from teacher checkpoint
    quantize     Build quantized export artifacts
    hw-profile   Show active hardware profile
    eval-harness Run evaluation with manifest harness
  improve      Self-improvement loop
  eval         Run evaluation suite
    benchmark-harness Run evaluation report; external scale comparison disabled/unverified
  audit        Run production-readiness audit checks
  full         Run full pipeline (tokenizer -> download -> train -> sft -> dpo -> improve -> eval)
  status       Show pipeline status
        """,
    )
    parser.add_argument(
        "command",
        choices=[
            "agent-plan", "agent-solve", "agent-verify",
            "tokenizer", "download", "download-safe", "download-core", "download-status", "token-manifest",
            "token-integrity", "deps", "hardware-validate", "gpu-fit-validate", "data-governance", "validate-real-path",
            "validate-short-run",
            "train-preflight", "deployment-info", "train", "sft", "dpo", "distill", "quantize", "hw-profile", "eval-harness",
            "improve", "eval", "benchmark-harness", "audit", "full", "status",
        ],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML/JSON config file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-env-overrides",
        action="store_true",
        help="Disable MYLLM__ environment variable config overrides.",
    )
    parser.add_argument(
        "--strict-config",
        action="store_true",
        help="Fail on unknown config sections or fields.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON for commands that support it.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="JSON report path for report-producing validation commands.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Agent task text for agent-plan/agent-solve/agent-verify.",
    )
    parser.add_argument(
        "--task-file",
        default=None,
        help="Path to a text file containing the agent task.",
    )
    parser.add_argument(
        "--file-hint",
        action="append",
        default=[],
        help="Relevant file path hint for agent commands. Can be repeated.",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="Structured verifier spec such as compileall:path.py or pytest:tests/test_x.py -q. Can be repeated.",
    )
    parser.add_argument(
        "--backend-script",
        default=None,
        help="Path to a scripted candidate JSON file for Phase 1 agent solve.",
    )

    args = parser.parse_args()

    try:
        apply_runtime_config(
            config_paths=args.config,
            include_env=not args.no_env_overrides,
            strict=args.strict_config,
        )
    except Exception as e:
        print(f"ERROR: failed to apply runtime config: {e}")
        sys.exit(1)

    OUTPUT_JSON = bool(args.json)
    REPORT_PATH = args.report_path

    check_dependencies(args.command)

    commands = {
        "agent-plan": lambda: run_agent_plan(args),
        "agent-solve": lambda: run_agent_solve(args),
        "agent-verify": lambda: run_agent_verify(args),
        "tokenizer": run_tokenizer,
        "download": run_download,
        "download-safe": run_download_safe,
        "download-core": run_download_core,
        "download-status": run_download_status,
        "token-manifest": run_token_manifest,
        "token-integrity": run_token_integrity,
        "deps": run_deps,
        "hardware-validate": run_hardware_validate,
        "gpu-fit-validate": run_gpu_fit_validate,
        "data-governance": run_data_governance,
        "validate-real-path": run_validate_real_path,
        "validate-short-run": run_validate_short_run,
        "train-preflight": run_train_preflight,
        "deployment-info": run_deployment_info,
        "train": run_train,
        "sft": run_sft,
        "dpo": run_dpo,
        "distill": run_distill,
        "quantize": run_quantize,
        "hw-profile": run_hw_profile,
        "eval-harness": run_eval_harness,
        "improve": run_improve,
        "eval": run_eval,
        "benchmark-harness": run_benchmark_harness,
        "audit": run_audit,
        "full": run_full,
        "status": show_status,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
