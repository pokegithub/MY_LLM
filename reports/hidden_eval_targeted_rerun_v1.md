# Hidden Eval Targeted Rerun Workflow v1

## Purpose

`hidden-eval-rerun-failed` reruns only selected hidden-eval seed items instead of rerunning the full seed set. It is intended for local debugging of known failed or blocked seeds such as `hidden_code_repair_001` or `hidden_retrieval_002`.

## Usage

Rerun all non-passed items from a previous public summary:

```powershell
python run.py hidden-eval-rerun-failed --previous-summary run_artifacts/hidden_eval_runs/<run_id>/summary.json
```

Rerun explicit seed IDs while still comparing against a previous public summary:

```powershell
python run.py hidden-eval-rerun-failed --previous-summary run_artifacts/hidden_eval_runs/<run_id>/summary.json --hidden-eval-seed-id hidden_code_repair_001 --hidden-eval-seed-id hidden_retrieval_002
```

The existing explicit seed path remains available:

```powershell
python run.py hidden-eval-seed-run --hidden-eval-seed-id hidden_code_repair_001 --hidden-eval-seed-id hidden_retrieval_002
```

## Delta Report

The targeted rerun writes `hidden_eval_targeted_rerun_delta_v1` with:

- previous summary path and run id
- new run id and new summary path
- rerun seed IDs
- previous and new status by seed
- improved, regressed, and unchanged counts
- previous passed count for the rerun subset
- new passed count for the rerun subset
- public-safe per-item route/status/verifier fields
- `private_leakage: false`
- `training_executed: false`
- `phase_b_started: false`
- `quality_claim: none`

## Privacy and Training Safety

The workflow uses only public hidden-eval summary fields for comparison and delegates execution to the existing hidden-eval runner. It does not expose private target answers, behavior assertions, or hidden expected outputs in public summaries. Hidden-eval traces remain excluded from SFT-positive training data.

## What It Does Not Prove

This workflow does not train, tune, download models, use cloud, start Phase B, run a full bakeoff, or prove model quality. It only makes failed-seed debugging cheaper and easier by reducing unnecessary full-suite reruns.
