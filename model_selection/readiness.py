"""Pre-bakeoff candidate readiness checks.

These helpers intentionally do not run the full bakeoff. They only verify that a
candidate slot has enough local/runtime/structured-output evidence to become a
serious bakeoff entrant later.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from agent.backend import BACKEND_KIND_LOCAL_TRANSFORMERS, backend_smoke_report
from agent.orchestrator import solve_task
from agent.types import TaskRequest
from config import agent_cfg


DEFAULT_SHORTLIST_PATH = "./model_selection/candidate_shortlist_v1.json"
READINESS_SMOKE_SCHEMA = "candidate_readiness_smoke_v1"
DEFAULT_CANDIDATE_ID = "open_dense_7b_candidate_slot"


@contextmanager
def _agent_overrides(**values: Any) -> Iterator[None]:
    old_values = {key: getattr(agent_cfg, key) for key in values}
    try:
        for key, value in values.items():
            setattr(agent_cfg, key, value)
        yield
    finally:
        for key, value in old_values.items():
            setattr(agent_cfg, key, value)


def load_candidate_shortlist(path: str = DEFAULT_SHORTLIST_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_candidate_shortlist(payload)
    return payload


def validate_candidate_shortlist(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "candidate_shortlist_v1":
        raise ValueError("candidate shortlist schema must be candidate_shortlist_v1")
    if payload.get("winner_selected") is not False:
        raise ValueError("candidate shortlist must not select a winner")
    if payload.get("moe_allowed") is not False:
        raise ValueError("candidate shortlist must keep MoE disabled")
    if payload.get("dense_only_required") is not True:
        raise ValueError("candidate shortlist must require dense-only candidates")
    groups = list(payload.get("candidate_groups", []) or [])
    if len(groups) != 3:
        raise ValueError("candidate shortlist must define exactly three candidate groups")
    ids = [str(item.get("candidate_id", "")) for item in groups]
    if len(set(ids)) != 3:
        raise ValueError("candidate shortlist candidate ids must be unique")
    if any(str(item.get("dense_or_moe")) == "moe" for item in groups):
        raise ValueError("candidate shortlist must not include MoE candidates")
    custom = next((item for item in groups if item.get("candidate_role") == "current_custom_base_candidate"), None)
    if not custom:
        raise ValueError("candidate shortlist must include the current custom base candidate")
    if custom.get("custom_base_qualification_status") != "requires_real_checkpoint":
        raise ValueError("custom base must require a real checkpoint before bakeoff entry")


def _find_candidate(payload: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for item in payload.get("candidate_groups", []) or []:
        if item.get("candidate_id") == candidate_id:
            return item
    raise ValueError(f"candidate_id not found in shortlist: {candidate_id}")


def _resolve_model_path(model_path: Optional[str], workspace_root: str) -> Optional[Path]:
    if not model_path:
        return None
    path = Path(str(model_path)).expanduser()
    if not path.is_absolute():
        path = Path(workspace_root).resolve() / path
    return path.resolve()


def _base_report(candidate: Mapping[str, Any], model_path: Optional[Path]) -> Dict[str, Any]:
    return {
        "schema": READINESS_SMOKE_SCHEMA,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_role": candidate.get("candidate_role"),
        "size_class": candidate.get("size_class"),
        "dense_or_moe": candidate.get("dense_or_moe"),
        "model_path": str(model_path) if model_path else None,
        "model_path_exists": bool(model_path.exists()) if model_path else False,
        "local_availability_status": candidate.get("local_availability_status"),
        "license_deployment_review_status": candidate.get("license_deployment_review_status"),
        "model_load_status": "not_attempted",
        "generation_status": "not_attempted",
        "structured_output_parse_status": "not_attempted",
        "schema_validation_status": "not_attempted",
        "candidate_valid": False,
        "verifier_status": "not_attempted",
        "smoke_level": "not_started",
        "failure_class": None,
        "failure_reason": None,
        "structured_output_readiness": "blocked_unverified",
        "proves_model_quality": False,
        "bakeoff_winner_claim": "none",
        "quality_claim": "none",
        "bakeoff_executed": False,
    }


def _smoke_level_from_backend_report(backend_report: Mapping[str, Any]) -> str:
    if backend_report.get("generation_status") == "failed":
        return "generation_failed"
    if backend_report.get("structured_output_parse_status") == "passed":
        return "candidate_valid_verifier_not_run"
    if backend_report.get("failure_class") == "schema_validation_failed":
        return "schema_validation_failed"
    if backend_report.get("failure_class") == "malformed_candidate_output":
        return "parse_failed"
    if backend_report.get("failure_class") == "model_load_failed":
        return "model_load_failed"
    return str(backend_report.get("smoke_level") or "blocked_unverified")


def _run_tiny_verifier_if_candidate_valid(
    *,
    model_path: Path,
    report_dir_root: str,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="candidate_readiness_") as workspace:
        module_path = Path(workspace) / "backend_smoke_target.py"
        test_path = Path(workspace) / "test_backend_smoke_target.py"
        module_path.write_text("def value():\n    return 1\n", encoding="utf-8")
        test_path.write_text(
            "import unittest\n"
            "from backend_smoke_target import value\n\n"
            "class SmokeTargetTests(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(value(), 2)\n",
            encoding="utf-8",
        )
        with _agent_overrides(
            backend_kind=BACKEND_KIND_LOCAL_TRANSFORMERS,
            backend_model_id_or_path=str(model_path),
            backend_local_files_only=True,
            backend_trust_remote_code=False,
            report_dir=os.path.join(report_dir_root, "agent_reports"),
        ):
            payload = solve_task(
                TaskRequest(
                    task_text="Change backend_smoke_target.py so value() returns 2.",
                    file_hints=("backend_smoke_target.py",),
                    checks=("unittest:discover -s . -p test_backend_smoke_target.py -v",),
                    workspace_root=workspace,
                )
            )
    return {
        "status": payload.get("status"),
        "verification_summary": (payload.get("verification") or {}).get("summary"),
        "quality_claim": payload.get("quality_claim", "none"),
        "blocked_reason": payload.get("blocked_reason"),
        "final_origin": payload.get("final_origin"),
    }


def candidate_readiness_smoke(
    *,
    candidate_id: str = DEFAULT_CANDIDATE_ID,
    shortlist_path: str = DEFAULT_SHORTLIST_PATH,
    workspace_root: str = ".",
) -> Dict[str, Any]:
    shortlist = load_candidate_shortlist(shortlist_path)
    candidate = _find_candidate(shortlist, candidate_id)
    model_path = _resolve_model_path(candidate.get("local_model_path"), workspace_root)
    report = _base_report(candidate, model_path)

    if candidate.get("custom_base_qualification_status") == "requires_real_checkpoint":
        report.update(
            {
                "smoke_level": "candidate_not_available",
                "failure_class": "candidate_not_available",
                "failure_reason": "current custom base requires a real usable checkpoint before readiness smoke",
            }
        )
        return report

    if model_path is None or not model_path.exists():
        report.update(
            {
                "smoke_level": "candidate_not_available",
                "failure_class": "candidate_not_available",
                "failure_reason": "candidate local model path is not available; this is not a quality failure",
            }
        )
        return report

    with tempfile.TemporaryDirectory(prefix="candidate_readiness_reports_") as report_root:
        with _agent_overrides(
            backend_kind=BACKEND_KIND_LOCAL_TRANSFORMERS,
            backend_model_id_or_path=str(model_path),
            backend_local_files_only=True,
            backend_trust_remote_code=False,
            report_dir=os.path.join(report_root, "agent_reports"),
        ):
            backend_report = backend_smoke_report(workspace_root=workspace_root)
            report.update(
                {
                    "model_load_status": backend_report.get("model_load_status"),
                    "generation_status": backend_report.get("generation_status"),
                    "structured_output_parse_status": backend_report.get("structured_output_parse_status"),
                    "schema_validation_status": backend_report.get("final_schema_validation_status"),
                    "candidate_valid": bool(backend_report.get("candidate_valid")),
                    "smoke_level": _smoke_level_from_backend_report(backend_report),
                    "failure_class": backend_report.get("failure_class"),
                    "failure_reason": backend_report.get("failure_reason"),
                    "backend_smoke": {
                        "schema": backend_report.get("schema"),
                        "structured_contract_version": backend_report.get("structured_contract_version"),
                        "generation_attempts": backend_report.get("generation_attempts"),
                        "malformed_retry_count": backend_report.get("malformed_retry_count"),
                        "raw_parse_status": backend_report.get("raw_parse_status"),
                        "normalization_attempted": backend_report.get("normalization_attempted"),
                        "normalization_applied": backend_report.get("normalization_applied"),
                        "normalization_kind": backend_report.get("normalization_kind"),
                        "normalization_rejected_reason": backend_report.get("normalization_rejected_reason"),
                        "final_parse_status": backend_report.get("final_parse_status"),
                        "schema_validation_status": backend_report.get("schema_validation_status"),
                        "unsafe_path_detected": backend_report.get("unsafe_path_detected"),
                        "quality_claim": backend_report.get("quality_claim"),
                    },
                }
            )
            if report["candidate_valid"]:
                tiny = _run_tiny_verifier_if_candidate_valid(
                    model_path=model_path,
                    report_dir_root=report_root,
                )
                report["tiny_verifier"] = tiny
                if tiny.get("status") == "verified_success":
                    report["verifier_status"] = "passed"
                    report["smoke_level"] = "candidate_valid_verifier_passed"
                else:
                    report["verifier_status"] = "failed"
                    report["smoke_level"] = "candidate_valid_verifier_failed"
                report["structured_output_readiness"] = "passed"
            elif report["smoke_level"] in {"parse_failed", "schema_validation_failed"}:
                report["structured_output_readiness"] = "failed"
                report["verifier_status"] = "blocked_candidate_invalid"
            else:
                report["structured_output_readiness"] = "blocked_unverified"
                report["verifier_status"] = "blocked_before_candidate"

    return report
