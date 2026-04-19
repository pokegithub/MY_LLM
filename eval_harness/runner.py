"""Wrapper around eval_suite with manifest + benchmark exclusion summary."""

import os
import json

from config import eval_cfg, train_cfg
from eval_suite import run_evaluation
from eval_harness.manifests import start_manifest, add_artifact, save_manifest
from eval_harness.contamination_checks import has_low_benchmark_exclusion_coverage


BENCHMARK_SOURCES = [
    "truthful_qa",
    "google/boolq",
    "allenai/ai2_arc",
    "winogrande",
    "openai/gsm8k",
]


def run(output_dir: str = "./eval_results"):
    manifest = start_manifest(output_dir, {"strict": eval_cfg.strict_real_benchmarks})

    weak_exclusion, protection_coverage = has_low_benchmark_exclusion_coverage(
        train_cfg.excluded_benchmark_sources,
        BENCHMARK_SOURCES,
        threshold=0.2,
    )

    results = run_evaluation(eval_cfg)
    report_path = os.path.join(output_dir, "eval_report.json")
    add_artifact(manifest, "eval_report", report_path)
    manifest["benchmark_exclusion_coverage"] = protection_coverage
    manifest["benchmark_exclusion_coverage_low"] = bool(weak_exclusion)
    manifest["contamination_protection_coverage"] = protection_coverage
    # Backward-compatible key used by existing downstream tooling.
    manifest["contamination_overlap"] = protection_coverage
    manifest["contamination_flag"] = bool(weak_exclusion)
    eval_metadata = results.get("metadata", {})
    manifest["eval_run_mode"] = eval_metadata.get("run_mode", "unknown")
    manifest["real_benchmark_evidence"] = bool(
        eval_metadata.get("real_benchmark_evidence", False)
    )
    manifest["eval_warnings"] = list(eval_metadata.get("warnings", []))

    mpath = save_manifest(manifest, output_dir)
    print(f"Eval harness manifest: {mpath}")

    # Keep a compact phase marker for downstream review.
    phase = {
        "phase": "eval_harness",
        "overall": results.get("scorecard", {}).get("Overall Practical Capability", None),
        "contamination_overlap": protection_coverage,
        "contamination_protection_coverage": protection_coverage,
        "benchmark_exclusion_coverage": protection_coverage,
        "benchmark_exclusion_coverage_low": bool(weak_exclusion),
        "eval_run_mode": eval_metadata.get("run_mode", "unknown"),
        "real_benchmark_evidence": bool(
            eval_metadata.get("real_benchmark_evidence", False)
        ),
        "contamination_flag": bool(weak_exclusion),
    }
    with open(os.path.join(output_dir, "phase_eval_harness.json"), "w", encoding="utf-8") as f:
        json.dump(phase, f, indent=2)


if __name__ == "__main__":
    run()
