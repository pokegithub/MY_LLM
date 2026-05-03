"""Integrity helpers for the private hidden-eval artifact set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


HIDDEN_EVAL_DIR = Path(__file__).resolve().parents[1] / "evals" / "hidden"
SPEC_JSON_PATH = HIDDEN_EVAL_DIR / "hidden_eval_spec_v1.json"
SPEC_MD_PATH = HIDDEN_EVAL_DIR / "hidden_eval_spec_v1.md"
SEED_SET_PATH = HIDDEN_EVAL_DIR / "hidden_eval_seed_set_v1.jsonl"
PRIVATE_TARGETS_PATH = HIDDEN_EVAL_DIR / "private" / "hidden_eval_private_targets_v1.json"
HOLDOUT_REGISTRY_PATH = (
    HIDDEN_EVAL_DIR / "hidden_eval_holdout_hash_registry_v1.json"
)

CATEGORY_NAMES = (
    "exact_symbolic_correctness",
    "abstention_need_tool_need_source",
    "source_grounded_truthfulness",
    "coding_patch_success",
    "coding_repair_loop_success",
    "retrieval_grounded_qa",
    "verifier_loop_kpis_latency_cost",
)
SPLIT_NAMES = ("frozen", "rotating")
EVALUATION_MODE_NAMES = (
    "exact_output",
    "exact_file_lookup",
    "decision_only",
    "answer_with_citations",
    "hidden_behavior_patch",
    "hidden_behavior_repair",
    "solve_report_audit",
)
REQUIRED_EVIDENCE_TYPES = (
    "exact_output",
    "abstention_or_tool_request",
    "citation_bundle",
    "hidden_behavior_report",
    "repair_report",
    "solve_report_metrics",
)
REQUIRED_SEED_FIELDS = (
    "id",
    "category",
    "task_text",
    "evaluation_mode",
    "required_evidence_type",
    "scoring_rule_ref",
    "holdout_metadata",
    "freshness_sensitive",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_json_payload(payload: Any) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def hash_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def load_hidden_eval_spec() -> Dict[str, Any]:
    return _load_json(SPEC_JSON_PATH)


def load_hidden_eval_seed_set() -> List[Dict[str, Any]]:
    return _load_jsonl(SEED_SET_PATH)


def load_hidden_eval_private_targets() -> Dict[str, Any]:
    return _load_json(PRIVATE_TARGETS_PATH)


def load_hidden_eval_holdout_registry() -> Dict[str, Any]:
    return _load_json(HOLDOUT_REGISTRY_PATH)


def validate_hidden_eval_assets() -> Dict[str, Any]:
    spec = load_hidden_eval_spec()
    seed_items = load_hidden_eval_seed_set()
    private_targets = load_hidden_eval_private_targets()
    registry = load_hidden_eval_holdout_registry()

    errors: List[str] = []
    categories = spec.get("categories", {})
    scoring_rules = spec.get("scoring_rules", {})
    registry_items = registry.get("items", [])
    registry_by_id = {item.get("id"): item for item in registry_items}

    if spec.get("schema") != "hidden_eval_spec_v1":
        errors.append("spec schema must be hidden_eval_spec_v1")
    if set(categories.keys()) != set(CATEGORY_NAMES):
        errors.append("spec category set does not match required category whitelist")
    if not scoring_rules:
        errors.append("spec scoring_rules must not be empty")

    ids = set()
    category_counts = {name: 0 for name in CATEGORY_NAMES}
    split_counts = {name: 0 for name in SPLIT_NAMES}

    for item in seed_items:
        missing = [field for field in REQUIRED_SEED_FIELDS if field not in item]
        if missing:
            errors.append(f"seed item {item.get('id', '<missing-id>')} missing fields: {missing}")
            continue

        item_id = str(item["id"])
        if item_id in ids:
            errors.append(f"duplicate seed item id: {item_id}")
        ids.add(item_id)

        category = item["category"]
        if category not in CATEGORY_NAMES:
            errors.append(f"seed item {item_id} has unknown category: {category}")
        else:
            category_counts[category] += 1

        mode = item["evaluation_mode"]
        if mode not in EVALUATION_MODE_NAMES:
            errors.append(f"seed item {item_id} has unknown evaluation_mode: {mode}")

        evidence = item["required_evidence_type"]
        if evidence not in REQUIRED_EVIDENCE_TYPES:
            errors.append(f"seed item {item_id} has unknown required_evidence_type: {evidence}")

        rule = item["scoring_rule_ref"]
        if rule not in scoring_rules:
            errors.append(f"seed item {item_id} references unknown scoring_rule_ref: {rule}")

        holdout = item["holdout_metadata"]
        split = holdout.get("split")
        if split not in SPLIT_NAMES:
            errors.append(f"seed item {item_id} has invalid split: {split}")
        else:
            split_counts[split] += 1

        if not holdout.get("training_exclusion_required", False):
            errors.append(f"seed item {item_id} must require training exclusion")

        private_ref = item.get("private_target_ref")
        if not private_ref:
            errors.append(f"seed item {item_id} is missing private_target_ref")
        elif private_ref not in private_targets:
            errors.append(f"seed item {item_id} references unknown private_target_ref: {private_ref}")

        registry_entry = registry_by_id.get(item_id)
        if registry_entry is None:
            errors.append(f"seed item {item_id} missing from holdout registry")
            continue

        expected_seed_hash = hash_json_payload(item)
        if registry_entry.get("seed_item_sha256") != expected_seed_hash:
            errors.append(f"seed hash mismatch for {item_id}")

        if registry_entry.get("category") != category:
            errors.append(f"registry category mismatch for {item_id}")
        if registry_entry.get("split") != split:
            errors.append(f"registry split mismatch for {item_id}")

        if private_ref in private_targets:
            expected_private_hash = hash_json_payload(private_targets[private_ref])
            if registry_entry.get("private_target_sha256") != expected_private_hash:
                errors.append(f"private target hash mismatch for {item_id}")

        expected_artifact_hashes = {}
        for rel_path in item.get("artifact_refs", []):
            artifact_path = HIDDEN_EVAL_DIR / rel_path
            if not artifact_path.exists():
                errors.append(f"artifact path missing for {item_id}: {rel_path}")
                continue
            expected_artifact_hashes[rel_path] = hash_file(artifact_path)

        recorded_artifact_hashes = {
            entry["path"]: entry["sha256"]
            for entry in registry_entry.get("artifact_hashes", [])
        }
        if recorded_artifact_hashes != expected_artifact_hashes:
            errors.append(f"artifact hash mismatch for {item_id}")

    if set(registry_by_id.keys()) != ids:
        errors.append("registry ids do not match seed item ids")

    if registry.get("split_summary") != split_counts:
        errors.append("registry split_summary does not match computed counts")
    if registry.get("category_summary") != category_counts:
        errors.append("registry category_summary does not match computed counts")

    if not all(category_counts.values()):
        missing_categories = [name for name, count in category_counts.items() if count == 0]
        errors.append(f"seed set does not cover all categories: {missing_categories}")

    if split_counts["frozen"] == 0 or split_counts["rotating"] == 0:
        errors.append("seed set must include both frozen and rotating items")

    if errors:
        raise ValueError("hidden eval asset validation failed: " + " | ".join(errors))

    return {
        "schema": "hidden_eval_asset_validation_v1",
        "spec_schema": spec["schema"],
        "seed_schema": registry.get("seed_schema"),
        "registry_schema": registry.get("schema"),
        "item_count": len(seed_items),
        "split_counts": split_counts,
        "category_counts": category_counts,
        "private_target_count": len(private_targets),
        "training_exclusion_required_for_all": True,
    }
