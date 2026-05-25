"""Narrow non-leaderboard comparison helpers for small backend smoke runs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _counts(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return summary.get("counts") or {}


def build_small_model_smoke_comparison(
    *,
    baseline_label: str,
    baseline_hidden_eval_summary: Mapping[str, Any],
    candidate_label: str,
    candidate_hidden_eval_summary: Mapping[str, Any],
    baseline_backend_smoke: Optional[Mapping[str, Any]] = None,
    candidate_backend_smoke: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare only observed smoke/eval status without selecting a winner."""
    baseline_counts = _counts(baseline_hidden_eval_summary)
    candidate_counts = _counts(candidate_hidden_eval_summary)
    return {
        "schema": "small_model_smoke_comparison_v1",
        "comparison_scope": "phase_a_small_model_smoke_only",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline": {
            "model_id_or_path": baseline_hidden_eval_summary.get("model_id_or_path"),
            "backend_kind": baseline_hidden_eval_summary.get("backend_kind"),
            "model_loaded": (baseline_backend_smoke or {}).get("model_load_status"),
            "generation_status": (baseline_backend_smoke or {}).get("generation_status"),
            "structured_output_parse_status": (baseline_backend_smoke or {}).get("structured_output_parse_status"),
            "candidate_valid": (baseline_backend_smoke or {}).get("candidate_valid"),
            "hidden_backend_routed": baseline_counts.get("backend_routed", 0),
            "hidden_verifier_reached": baseline_counts.get("verifier_reached", 0),
            "hidden_verifier_passed": baseline_counts.get("verifier_passed", 0),
            "hidden_blocked": baseline_counts.get("blocked", 0),
            "hidden_failed": baseline_counts.get("failed", 0),
        },
        "candidate": {
            "model_id_or_path": candidate_hidden_eval_summary.get("model_id_or_path"),
            "backend_kind": candidate_hidden_eval_summary.get("backend_kind"),
            "model_loaded": (candidate_backend_smoke or {}).get("model_load_status"),
            "generation_status": (candidate_backend_smoke or {}).get("generation_status"),
            "structured_output_parse_status": (candidate_backend_smoke or {}).get("structured_output_parse_status"),
            "candidate_valid": (candidate_backend_smoke or {}).get("candidate_valid"),
            "hidden_backend_routed": candidate_counts.get("backend_routed", 0),
            "hidden_verifier_reached": candidate_counts.get("verifier_reached", 0),
            "hidden_verifier_passed": candidate_counts.get("verifier_passed", 0),
            "hidden_blocked": candidate_counts.get("blocked", 0),
            "hidden_failed": candidate_counts.get("failed", 0),
        },
        "winner_selected": None,
        "bakeoff_executed": False,
        "quality_claim": "none",
        "proves_model_quality": False,
        "notes": [
            "Exact-tool results are not model intelligence evidence.",
            "Verifier status on a tiny hidden seed subset is not general model quality.",
            "Unsupported retrieval/source-grounded categories are not model failures.",
        ],
    }
