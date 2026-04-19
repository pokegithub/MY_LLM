"""Automated benchmark harness with 1x/10x/100x comparison projections."""

from __future__ import annotations

import json
import os
import time
from csv import DictWriter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from config import EvalConfig, eval_cfg
from eval_suite import run_evaluation


@dataclass(frozen=True)
class ScaleBandTarget:
    scale: str
    models: List[str]
    expected_overall_min: float
    expected_overall_max: float


SCALE_BANDS: List[ScaleBandTarget] = [
    ScaleBandTarget(
        scale="1x",
        models=["TinyLlama-1.1B", "Qwen-2.5-0.5B"],
        expected_overall_min=3.5,
        expected_overall_max=6.5,
    ),
    ScaleBandTarget(
        scale="10x",
        models=["Llama-3.2-8B", "Mistral-Nemo"],
        expected_overall_min=6.0,
        expected_overall_max=8.5,
    ),
    ScaleBandTarget(
        scale="100x",
        models=["GPT-4o", "Claude 3.5 Sonnet"],
        expected_overall_min=8.0,
        expected_overall_max=9.8,
    ),
]

REPORT_CARD_METRICS: List[str] = [
    "Reasoning Depth",
    "English Understanding",
    "Context Understanding",
    "General Knowledge",
    "Throughput (TPS)",
    "Token Efficiency",
    "Task Completion",
    "Answer Relevancy",
    "Coding Reliability",
    "Factual Robustness",
    "Instruction Following",
    "Consistency/Stability",
    "Long-Context Handling",
    "Hallucination Resistance",
    "Training Speed (Proxy)",
    "Latency Efficiency",
    "Fine-Tune Agility",
    "Deployment Practicality",
]


def build_report_card_18(scorecard: Dict[str, Any]) -> Dict[str, Any]:
    metrics = []
    for name in REPORT_CARD_METRICS:
        metrics.append({
            "metric": name,
            "score": round(float(scorecard.get(name, 0.0)), 2),
        })

    overall = round(
        sum(item["score"] for item in metrics) / max(len(metrics), 1),
        2,
    )
    return {
        "metric_count": len(metrics),
        "metrics": metrics,
        "overall": overall,
    }


def build_scale_comparison(overall: float) -> Dict[str, Any]:
    model_overall = round(float(overall), 2)

    rows = []
    for band in SCALE_BANDS:
        lower_gap = round(model_overall - band.expected_overall_min, 2)
        upper_gap = round(model_overall - band.expected_overall_max, 2)
        status = "below"
        if band.expected_overall_min <= model_overall <= band.expected_overall_max:
            status = "within"
        elif model_overall > band.expected_overall_max:
            status = "above"

        rows.append(
            {
                "scale": band.scale,
                "models": list(band.models),
                "target_range": [band.expected_overall_min, band.expected_overall_max],
                "model_overall": model_overall,
                "gap_to_range_min": lower_gap,
                "gap_to_range_max": upper_gap,
                "status": status,
            }
        )

    return {
        "overall": model_overall,
        "bands": rows,
    }


def write_scale_csv(path: str, comparison: Dict[str, Any]) -> None:
    rows = comparison.get("bands", [])
    fields = [
        "scale",
        "models",
        "target_range_min",
        "target_range_max",
        "model_overall",
        "gap_to_range_min",
        "gap_to_range_max",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            target_min, target_max = row.get("target_range", [0.0, 0.0])
            writer.writerow(
                {
                    "scale": row.get("scale", ""),
                    "models": " | ".join(row.get("models", [])),
                    "target_range_min": target_min,
                    "target_range_max": target_max,
                    "model_overall": row.get("model_overall", 0.0),
                    "gap_to_range_min": row.get("gap_to_range_min", 0.0),
                    "gap_to_range_max": row.get("gap_to_range_max", 0.0),
                    "status": row.get("status", ""),
                }
            )


def write_summary_markdown(
    path: str,
    report_card: Dict[str, Any],
    comparison: Dict[str, Any],
) -> None:
    lines = [
        "# Benchmark Harness Summary",
        "",
        f"Overall Practical Capability: **{report_card.get('overall', 0.0):.2f} / 10**",
        "",
        "## 18-Metric Report Card",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for item in report_card.get("metrics", []):
        lines.append(f"| {item['metric']} | {item['score']:.2f} |")

    lines.extend([
        "",
        "## Scale Comparison (1x/10x/100x)",
        "",
        "| Scale | Target Min | Target Max | Model Overall | Status |",
        "|---|---:|---:|---:|---|",
    ])
    for row in comparison.get("bands", []):
        target_min, target_max = row.get("target_range", [0.0, 0.0])
        lines.append(
            "| "
            f"{row.get('scale', '')} | "
            f"{target_min:.2f} | "
            f"{target_max:.2f} | "
            f"{row.get('model_overall', 0.0):.2f} | "
            f"{row.get('status', '')} |"
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_benchmark_harness(
    cfg: EvalConfig = eval_cfg,
    output_dir: str = "./eval_results",
) -> Dict[str, Any]:
    """Run evaluation and emit a structured comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    results = run_evaluation(cfg)
    scorecard = results.get("scorecard", {})
    report_card = build_report_card_18(scorecard)
    comparison = build_scale_comparison(report_card["overall"])

    report_card_path = os.path.join(output_dir, "benchmark_harness_report_card_18.json")
    scale_csv_path = os.path.join(output_dir, "benchmark_harness_scale_comparison.csv")
    summary_md_path = os.path.join(output_dir, "benchmark_harness_summary.md")

    with open(report_card_path, "w", encoding="utf-8") as handle:
        json.dump(report_card, handle, indent=2)

    write_scale_csv(scale_csv_path, comparison)
    write_summary_markdown(summary_md_path, report_card, comparison)

    report = {
        "generated_at": int(time.time()),
        "elapsed_sec": round(time.time() - started, 3),
        "scale_targets": [asdict(band) for band in SCALE_BANDS],
        "comparison": comparison,
        "scorecard": scorecard,
        "report_card_18": report_card,
        "operational": results.get("operational", {}),
        "metadata": results.get("metadata", {}),
        "artifacts": {
            "report_card_18": os.path.abspath(report_card_path),
            "scale_csv": os.path.abspath(scale_csv_path),
            "summary_markdown": os.path.abspath(summary_md_path),
        },
    }

    out_path = os.path.join(output_dir, "benchmark_harness_report.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Benchmark harness report: {out_path}")
    print(f"Benchmark report card: {report_card_path}")
    print(f"Benchmark scale table: {scale_csv_path}")
    print(f"Benchmark summary md: {summary_md_path}")
    return report


if __name__ == "__main__":
    run_benchmark_harness()
