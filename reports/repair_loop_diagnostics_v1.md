# Repair Loop Diagnostics v1

## Purpose

This report documents a narrow Phase A maintenance change: failed coding-agent repair attempts now produce clearer diagnostics while preserving verifier authority, rollback behavior, hidden-eval privacy, and trajectory-quality filtering.

## Diagnostics Added

Failed or attempted repair-loop reports may now include `failure_diagnostics` with:

- `final_failure_class`
- `attempt_count`
- `repair_budget_exhausted`
- `verifier_failure_type`
- `all_attempts_failed_same_check`
- `distinct_failure_signatures`
- `last_verifier_summary`
- `failure_is_model_or_patch_behavior`
- `infrastructure_failure_detected`
- `hidden_or_fixture_workspace_detected`
- `training_use_allowed: false`
- `no_winning_candidate`
- `all_failed_edits_rolled_back`
- `per_attempt_evidence_paths_distinct`
- per-attempt candidate/verifier evidence references

Verifier stdout/stderr paths are now scoped under per-attempt directories for coding solve attempts, so repeated failures do not overwrite earlier verifier evidence.

## What This Does Not Prove

- It does not prove model quality.
- It does not make failed repair traces training data.
- It does not create SFT-positive eligibility.
- It does not start Phase B.
- It does not change hidden targets, verifier success criteria, or backend behavior.

## Training and Hidden-Eval Safety

Failed hidden-eval repair traces remain non-training-positive. Hidden/private expected answers are not added to prompts, docs, public summaries, or training data. Public hidden-eval summaries expose only safe scalar diagnostics such as `failure_category`, `repair_budget_exhausted`, `training_use_allowed: false`, and privacy flags.

## Verifier Authority

The verifier remains authoritative. A clearer diagnostic report does not convert verifier failure into success and does not weaken schema validation, output normalization, or trajectory-quality filtering.
