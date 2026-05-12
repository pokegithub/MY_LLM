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

## Optional Backend Dependencies

Backend runtime dependencies are declared separately in:

`requirements-backend.txt`

They are backend-specific. Missing backend dependencies must make backend smoke fail clearly, but they must not change no-backend fail-closed behavior or deterministic exact-task routing.

## Provision Tiny Smoke Model

Use the explicit provisioning command only when you intend to download a tiny smoke model:

```powershell
.\.venv\Scripts\python.exe run.py agent-backend-provision-tiny-model
```

By default the command allows only:

- `hf-internal-testing/tiny-random-gpt2`
- `sshleifer/tiny-gpt2`

It writes to:

`./run_artifacts/local_models/tiny-transformers-smoke`

This download is for backend smoke only. It is not bakeoff evidence, not a useful coding model, and not training data.

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

Smoke levels:

- `runtime_unavailable`: backend runtime dependency is missing or cannot be imported.
- `model_missing`: configured local-only model path is absent or incomplete.
- `model_loaded_generation_failed`: model loaded but generation failed.
- `model_loaded_generation_succeeded_parse_failed`: model loaded and generated text, but strict candidate parsing failed.
- `model_loaded_generation_succeeded_parse_passed`: model loaded, generated text, and produced a parseable candidate.

Every outcome must keep `proves_real_coding_ability: false` and `quality_claim: none` unless a separate verified solve report proves a specific toy task passed its verifier.
