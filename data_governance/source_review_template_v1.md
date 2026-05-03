# Source Review Template v1

Purpose: repo-side source review evidence for training readiness. This template is not legal advice, not legal clearance, and not permission to start serious training.

Use this template for a small manually reviewed candidate set only. Keep `legal_clearance_claim: none`.

## Required identity fields

- `source_id`:
- `subset`:
- `source_category`:
- `source_domain`:

## Documented metadata

- `provenance_evidence`:
  - Use source URLs or other concrete references.
- `declared_license`:
  - Use `unknown` if not manually documented from a concrete source.
- `license_evidence_source`:
  - Allowed values:
    - `unknown`
    - `manual_repo_metadata`
    - `manual_dataset_card_metadata`
    - `manual_upstream_legal_page`
    - `manual_dataset_card_and_upstream_legal_page`
- `review_basis`:
  - Allowed values:
    - `not_reviewed`
    - `manual_source_page_review`
    - `manual_dataset_card_review`
    - `manual_repo_policy_decision`

## Benchmark / holdout risk

- `benchmark_risk_status`:
  - Allowed values:
    - `not_flagged`
    - `benchmark_excluded`
    - `benchmark_adjacent_needs_review`

## Repo policy decision

- `repo_policy_status`:
  - Allowed values:
    - `unspecified`
    - `excluded`
    - `blocked_pending_review`
    - `allowed_by_repo_policy`
    - `restricted`
- `governance_classification`:
  - Allowed values:
    - `unknown`
    - `needs_manual_review`
    - `excluded_from_training`
    - `documented_but_unreviewed`
    - `policy_allowed_but_not_legally_cleared`
- `training_blocker_level`:
  - Allowed values:
    - `hard_blocker`
    - `caution`
    - `informational`

## Training-readiness intent

- `intended_training_role`:
  - Allowed values:
    - `none`
    - `pending_review_only`
    - `reference_only`
    - `instruction_sft_candidate`
    - `preference_optimization_only`
- `exclusion_rationale`:
  - Required when `repo_policy_status` is `excluded`, `blocked_pending_review`, or `restricted`.

## Truthfulness fields

- `notes`:
  - Keep caveats explicit. Unknown means unknown.
- `legal_clearance_claim`:
  - Allowed value:
    - `none`

## Review discipline

- Separate documented metadata from repo policy.
- `allowed_by_repo_policy` does not imply legal approval.
- A documented license string does not imply legal clearance.
- Benchmark-excluded or hidden-eval material must stay out of training.
- If the strongest honest result is "still blocked", record that directly.
