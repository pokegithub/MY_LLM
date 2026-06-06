"""Low-risk status/dependency command wrappers for run.py."""

import os
import sys

from core.dependency_checks import check_requirement_file
from cli.output import print_banner


BACKEND_REQUIREMENTS_PATH = "./requirements-backend.txt"

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


def collect_backend_dependency_checks(requirements_path: str = BACKEND_REQUIREMENTS_PATH):
    """Collect optional backend dependency drift without gating non-backend commands."""
    report = check_requirement_file(requirements_path)
    report["optional"] = True
    report["purpose"] = "local_transformers_backend"
    return report


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


def _print_backend_dependency_summary(report):
    print("\nOPTIONAL BACKEND DEPENDENCIES")
    print(f"  requirements: {os.path.abspath(report['path'])}")
    if not report["exists"]:
        print("  [UNVERIFIED] optional backend requirement file is missing")
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
    print("  gates       : backend-specific only; normal repo commands remain fail-closed if unavailable")


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
    backend_dependency_report = collect_backend_dependency_checks()
    return {
        "schema": "pipeline_status_v2",
        "status_checks": collect_status_checks(),
        "dependency_environment": dependency_report,
        "optional_backend_dependency_environment": backend_dependency_report,
        "next_action": _next_status_action(dependency_report["ok"]),
        "training_quality_claim": "none",
    }


def show_status(output_json: bool = False, emit_json=None):
    """Show current pipeline status."""
    report = build_status_report()
    if output_json:
        emit_json(report)
        return

    print_banner("PIPELINE STATUS")

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
    _print_backend_dependency_summary(report["optional_backend_dependency_environment"])

    next_action = report["next_action"]
    print(f"\n  NEXT: {next_action['command']}")
    if next_action["command"] == "python run.py token-manifest":
        print("  Training remains NOT READY until train-preflight passes.")
    elif not dependency_report["ok"]:
        print("  Do not start training from this environment yet.")
    elif next_action["command"] == "python run.py train-preflight":
        print("  Only run python run.py train after preflight passes.")


def run_deps(output_json: bool = False, emit_json=None):
    """Show declared dependency satisfaction."""
    report = collect_dependency_checks()
    backend_report = collect_backend_dependency_checks()
    report["optional_backend_dependency_environment"] = backend_report
    if output_json:
        emit_json(report)
        if not report["ok"]:
            sys.exit(1)
        return

    _print_dependency_summary(report)
    _print_backend_dependency_summary(backend_report)
    if not report["ok"]:
        print("\nDEPENDENCY CHECK FAILED")
        sys.exit(1)
    print("\nDEPENDENCY CHECK PASSED")

