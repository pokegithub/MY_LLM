# Repair Loop Prompt Critique v1

## Purpose

This report records a narrow Phase A repair-loop prompt/critique improvement. The goal is to make verifier failure feedback clearer for a later repair attempt without changing verifier authority, success criteria, rollback behavior, training eligibility, hidden-eval privacy, or backend runtime semantics.

## What Improved

- Repair attempts now receive an enriched critique payload before backend generation.
- The payload includes the verifier failure class, failing check summary, touched file paths, previous candidate summary, retry-budget context, and explicit instructions not to repeat failed behavior.
- The repair feedback reminds the backend that the verifier remains final authority and that the structured candidate contract still applies.
- Failure diagnostics now include safe scalar fields:
  - `repeated_failure_signature_detected`
  - `repeated_failed_behavior_warning`
  - `repair_prompt_includes_verifier_feedback`
  - `repair_prompt_includes_failure_class`
- Hidden-eval public summaries expose only safe scalar diagnostic flags, not hidden target answers or repair prompt text.

## What Did Not Change

- Verifier pass/fail authority did not change.
- Candidate schema validation did not change.
- Workspace rollback behavior did not change.
- Failed edits remain rolled back.
- Hidden private target contents were not read or exposed.
- Failed traces remain `training_use_allowed: false`.
- `quality_claim` remains `none` unless the verifier actually passes.
- SFT-positive eligibility remains controlled by the trajectory-quality audit.

## Why This Is Not Training

No model weights are changed. No SFT, DPO, RLVR, distillation, or replay learning is run. The change only improves the diagnostic text passed into future repair candidate generation and the safe scalar diagnostics written to reports.

## Why This Does Not Prove Model Quality

Better repair feedback can improve debuggability, but it is not model-quality evidence by itself. A model still has to produce a valid candidate, the candidate still has to pass schema/path checks, and the verifier still has to pass before any success can be claimed.

## Hidden-Answer Safety

The implementation does not inspect hidden private expected answers and does not put hidden private target content into public summaries. Synthetic public tests assert that repair prompts include verifier feedback while avoiding private-answer leakage markers.
