# Backend Smoke Execution Guide v1

Purpose: run the first local Transformers backend smoke without turning it into a bakeoff, training run, or coding-quality claim.

## Scope

This guide is only for `local_transformers_in_process` smoke execution.

It checks:

- explicit backend configuration
- local-only model loading
- candidate-generation call execution
- strict JSON candidate parsing
- truthful failure classification

It does not check:

- 7B/14B model quality
- hidden-eval performance
- production serving readiness
- SFT-positive trajectory usefulness
- retrieval or training behavior

## Tiny Local Model Requirement

Provide a small local Transformers-compatible causal language model directory, then override:

`agent.backend_model_id_or_path`

The example config points at:

`./run_artifacts/local_models/tiny-transformers-smoke`

That directory is intentionally not created by default. A smoke model must be supplied manually or produced by a separate, explicit setup step. The backend uses `backend_local_files_only: true` in the example so the smoke command does not auto-download model artifacts.

## Example Command

```powershell
.\.venv\Scripts\python.exe run.py agent-backend-smoke --config configs\backend_smoke_local_example.json
```

## Truthful Outcomes

- `backend_not_configured`: backend kind or model path is missing.
- `backend_unavailable`: required local runtime, such as `transformers`, is unavailable.
- `model_load_failed`: runtime is present but local model artifacts cannot be loaded.
- `malformed_candidate_output`: model loaded but did not return strict JSON.
- `schema_validation_failed`: JSON parsed but did not match the `Candidate` / `FileEdit` contract.
- `candidate_generated`: one candidate parsed successfully; this is still not proof of coding ability.

Every outcome must keep `proves_real_coding_ability: false` and `quality_claim: none` unless a separate verified solve report proves a specific toy task passed its verifier.
