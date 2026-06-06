# CLI Command Inventory Snapshot v1

## Purpose

This report records the frozen `run.py` command surface before any future source extraction or CLI boundary refactor. It exists to make future refactors safer by detecting accidental command removals, renames, dispatch drift, or help-surface omissions.

No commands were moved. No handlers were extracted. No runtime behavior changed.

## Snapshot Summary

- Commands frozen: 51.
- Fixture path: `tests/fixtures/cli_command_inventory_v1.json`.
- Test path: `tests/test_cli_command_inventory.py`.
- Parser/help and dispatch command names are expected to match exactly.
- High-risk commands are detected by inventory tests but not executed.
- Cloud/provisioning/download commands are detected by inventory tests but not executed.

## Command Categories

Safe local/status commands:

- `status`
- `deps`
- `compile-source`
- `audit`
- `deployment-info`
- `hw-profile`

Retrieval and repo-assist commands:

- `retrieval-index-build`
- `retrieval-search`
- `retrieval-citation-check`
- `retrieval-answer`
- `repo-assist`
- `repo-assist-eval`

Hidden-eval commands:

- `hidden-eval-seed-run`
- `hidden-eval-rerun-failed`

Backend/candidate commands:

- `agent-backend-smoke`
- `agent-backend-provision-tiny-model`
- `agent-backend-provision-small-candidate`
- `candidate-readiness-smoke`

Data/governance commands:

- `tokenizer`
- `download`
- `download-safe`
- `download-core`
- `download-status`
- `token-manifest`
- `token-integrity`
- `hardware-validate`
- `gpu-fit-validate`
- `data-governance`
- `validate-real-path`
- `validate-short-run`

Training/high-risk commands:

- `train`
- `sft`
- `dpo`
- `distill`
- `improve`
- `full`

## Test Scope

The snapshot tests:

- parse `run.py` command choices without importing command handlers;
- parse the dispatch dictionary keys without calling handlers;
- run only `python run.py --help`;
- assert the fixture, parser choices, dispatch keys, and help text agree;
- assert high-risk and provisioning/download commands are present but not executed.

The tests do not require GPU, internet, cloud credentials, model downloads, training data changes, or checkpoint files.

## Truth Limits

- This snapshot does not prove command behavior correctness.
- This snapshot does not prove model quality.
- This snapshot does not start Phase B.
- This snapshot does not make training ready.
- This snapshot is a pre-refactor guardrail only.

## Verification

Verification results:

| Command | Result | Notes |
| --- | --- | --- |
| `python -m pytest tests/test_cli_command_inventory.py -q` | Passed | 8 passed. |
| `python run.py status` | Passed | No training readiness claim. |
| `python run.py deps` | Passed | Core deps 7/0; optional backend deps 2/0. |
| `python run.py compile-source` | Passed | `scanned: 103`, tracked-source-only scope. |
| `python run.py repo-assist-eval` | Passed | `23/23`, `uses_web: false`, `uses_model_backend: false`. |
| `python run.py hidden-eval-seed-run` | Passed | 17 attempted, 13 passed, 4 blocked, `private_leakage: false`, `quality_claim: none`. |
| `python run.py trajectory-quality-audit` | Passed | `scanned: 1395`, `sft_positive: 0`, `rejected_positive: 1395`. |
| `python -m pytest tests -q` | Passed | 245 passed, 1 skipped. |
| `python -m compileall -q .` | Passed | No output. |
| `git diff --check` | Passed | No whitespace errors. |
| `git status --short` | Passed | Only report artifacts and the new test/fixture are untracked. |
