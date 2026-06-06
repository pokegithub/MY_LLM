# Reports Index v1

## Purpose

This index identifies the current Phase A evidence reports, supporting diagnostics, future cloud plans, and historical status snapshots. It is a navigation aid for handoff. It is not a runtime input, training dataset, model-quality claim, or Phase B start marker.

## Canonical Current Reports

| Report Path | Category | Status | Why It Matters | Current Truth Limits |
| --- | --- | --- | --- | --- |
| `reports/phase_a_final_checkpoint_v1.md` | Phase A checkpoint | Canonical current handoff evidence | Summarizes the stabilized Phase A state, including local RTX 2050 direction, retrieval/repo-assist scope, and no-training posture. | Does not start Phase B, prove model quality, or make training ready. |
| `reports/professional_repo_audit_v1.md` | Repo cleanliness audit | Canonical current cleanup evidence | Rates the repo at 72/100 overall maturity and identifies professionalization risks, artifact hygiene issues, and next cleanup passes. | Audit only; no behavior changes, deletions, or refactors were performed. |
| `reports/pre_cloud_batch_plan_v1.md` | Future cloud plan | Canonical future-work checklist | Defines the manual, guarded plan for a possible future 7B cloud batch. | Planning only; cloud is deferred, training is forbidden, and Phase B has not started. |
| `reports/cloud_qwen2_5_coder_7b_evidence_v1.md` | Cloud evidence snapshot | Canonical supporting evidence for 7B cloud smoke | Records synced evidence from the stopped Lightning AI Qwen2.5-Coder-7B smoke/hidden-eval run. | Not a broad model-quality claim, not training readiness, and not SFT-positive evidence. |

## Supporting Diagnostic Reports

| Report Path | Category | Status | Why It Matters | Current Truth Limits |
| --- | --- | --- | --- | --- |
| `reports/repair_loop_diagnostics_v1.md` | Repair-loop diagnostics | Supporting diagnostic | Documents clearer failed-attempt verifier evidence and rollback-safe repair diagnostics. | Does not weaken verifier authority or convert failures into training-positive traces. |
| `reports/repair_loop_prompt_critique_v1.md` | Repair-loop prompt critique | Supporting diagnostic | Documents improved repair prompt feedback using public/synthetic diagnostics only. | Does not use hidden answers, train, or prove model quality. |
| `reports/hidden_eval_targeted_rerun_v1.md` | Targeted hidden-eval workflow | Supporting workflow doc | Explains selected failed/non-passed hidden seed reruns and public-safe delta reporting. | Does not expose hidden private targets and does not make traces SFT-positive. |
| `reports/post_repair_targeted_rerun_checkpoint_v1.md` | Post-repair targeted rerun checkpoint | Supporting status snapshot | Records the targeted rerun checkpoint after repair-loop diagnostics and contradiction handling. | Snapshot only; not training readiness and not Phase B evidence. |

## Machine-Readable Companions

JSON companions are status snapshots for review and automation checks only. They are not runtime inputs unless a specific command explicitly documents that usage.

| JSON Path | Companion For | Status |
| --- | --- | --- |
| `reports/cloud_qwen2_5_coder_7b_evidence_v1.json` | `reports/cloud_qwen2_5_coder_7b_evidence_v1.md` | Tracked cloud evidence snapshot. |
| `reports/pre_cloud_batch_plan_v1.json` | `reports/pre_cloud_batch_plan_v1.md` | Accepted future cloud-plan snapshot. |
| `reports/professional_repo_audit_v1.json` | `reports/professional_repo_audit_v1.md` | Accepted audit summary snapshot. |
| `reports/post_repair_targeted_rerun_checkpoint_v1.json` | `reports/post_repair_targeted_rerun_checkpoint_v1.md` | Accepted targeted-rerun checkpoint snapshot. |

## Report Lifecycle Policy

- Canonical reports are current handoff evidence and should remain easy to find.
- Supporting reports are diagnostic/design evidence that explain why current guardrails and workflows exist.
- Historical reports are retained for traceability but should not be treated as the latest state without checking this index and `SYSTEM_MAP.md`.
- Generated run reports belong under `run_artifacts/` and should not be committed wholesale.
- Model artifacts, checkpoints, caches, `.venv`, `data_cache/`, `quantized/`, and generated cloud outputs must never be tracked.
- Cloud run artifacts should be summarized in reviewed reports, not committed wholesale.

## Current Truth State

- Phase A remains active.
- Phase B has not started.
- SFT-positive trajectory eligibility remains 0.
- Training, SFT, DPO, RL/RLVR, and distillation are not ready and have not started.
- Model quality is not broadly proven.
- Cloud is deferred until manually approved.
- Future cloud work should use `reports/pre_cloud_batch_plan_v1.md`.
- Hidden eval traces are not SFT-positive training data.

## Handoff Reading Order

1. `SYSTEM_MAP.md`
2. `reports/index_v1.md`
3. `reports/phase_a_final_checkpoint_v1.md`
4. `reports/professional_repo_audit_v1.md`
5. `reports/pre_cloud_batch_plan_v1.md`
6. `reports/cloud_qwen2_5_coder_7b_evidence_v1.md`
7. Relevant diagnostic reports:
   - `reports/repair_loop_diagnostics_v1.md`
   - `reports/repair_loop_prompt_critique_v1.md`
   - `reports/hidden_eval_targeted_rerun_v1.md`
   - `reports/post_repair_targeted_rerun_checkpoint_v1.md`

