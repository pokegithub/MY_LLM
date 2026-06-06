"""Low-risk compile-source command wrapper for run.py."""

import subprocess
import sys

from cli.output import format_bool, print_banner


SKIPPED_GENERATED_ROOTS = [
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "run_artifacts",
    "data_cache",
    "tokenizer_data",
    "checkpoints",
    "sft_checkpoints",
    "dpo_checkpoints",
    "distill_checkpoints",
    "improved_checkpoints",
    "quantized",
    "eval_results",
]


def run_compile_source(output_json: bool = False, emit_json=None):
    """Compile tracked source-like Python files without scanning generated artifact roots."""
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print("ERROR: git ls-files failed")
        if result.stderr:
            print(result.stderr.strip())
        sys.exit(1)

    tracked_paths = [
        item.strip()
        for item in result.stdout.splitlines()
        if item.strip()
    ]
    scanned = []
    skipped = []
    failures = []
    for rel_path in tracked_paths:
        normalized = rel_path.replace("\\", "/")
        if any(
            normalized == root or normalized.startswith(root.rstrip("/") + "/")
            for root in SKIPPED_GENERATED_ROOTS
        ):
            skipped.append(normalized)
            continue
        try:
            with open(rel_path, "rb") as handle:
                compile(handle.read(), rel_path, "exec")
            scanned.append(normalized)
        except Exception as exc:
            failures.append({"path": normalized, "error": str(exc)})

    payload = {
        "schema": "source_compile_report_v1",
        "ok": not failures,
        "scope": "tracked_python_sources_only",
        "full_compileall_replacement": False,
        "scanned_count": len(scanned),
        "skipped_count": len(skipped),
        "skipped_generated_roots": SKIPPED_GENERATED_ROOTS,
        "failures": failures,
        "behavior_changed": False,
    }
    if output_json:
        emit_json(payload)
    else:
        print_banner("SOURCE COMPILE")
        print(f"  scope                     : {payload['scope']}")
        print(f"  full_compileall_replacement: {format_bool(payload['full_compileall_replacement'])}")
        print(f"  scanned                   : {payload['scanned_count']}")
        print(f"  skipped                   : {payload['skipped_count']}")
        print(f"  skipped_roots             : {', '.join(SKIPPED_GENERATED_ROOTS)}")
        print(f"  ok                        : {format_bool(payload['ok'])}")
        for failure in failures[:10]:
            print(f"  failure                   : {failure['path']} :: {failure['error']}")
        print("=" * 60)
    if failures:
        sys.exit(1)

