# CLI Low-Risk Command Contracts v1

## Purpose

This report records the test package that locks observable CLI/JSON behavior for the lowest-risk future extraction candidates: `status`, `deps`, and `compile-source`.

No command handlers were moved. Parser choices and dispatch keys were not changed. Runtime behavior was not changed.

## Commands Covered

- `status`
- `deps`
- `compile-source`

## JSON Contracts Locked

The new tests assert stable top-level JSON keys for:

- `python run.py --json status`
- `python run.py --json deps`
- `python run.py --json compile-source`

Truthfulness expectations locked:

- `status.training_quality_claim == "none"`
- `status.next_action.ready_for_training` is false when present
- optional backend dependencies remain separate from broad training readiness proof
- `compile-source.scope == "tracked_python_sources_only"`
- `compile-source.full_compileall_replacement == false`
- `compile-source.behavior_changed == false`

## Text Markers Locked

The new tests assert stable critical substrings only, not fragile full formatting:

- `PIPELINE STATUS`
- `DEPENDENCY ENVIRONMENT`
- `OPTIONAL BACKEND DEPENDENCIES`
- `NEXT:`
- `DEPENDENCY CHECK PASSED`
- `SOURCE COMPILE`
- `tracked_python_sources_only`
- `full_compileall_replacement`
- `ok`

## What Is Not Tested

- Full whitespace/text snapshots.
- Training behavior.
- Cloud behavior.
- Model quality.
- Hidden-eval internals.
- Retrieval/citation internals.
- Backend parser or normalization behavior.

## Truth State

- Runtime behavior changed: no.
- Command handlers moved: no.
- Command inventory changed: no, still 51.
- Cloud used: no.
- Training started: no.
- Phase B started: no.
- Model quality claimed: no.
- SFT-positive remains 0.

## Verification

Verification results:

| Command | Result | Notes |
| --- | --- | --- |
| `python -m pytest tests/test_cli_command_inventory.py -q` | Passed | 8 passed. |
| `python -m pytest tests/test_cli_output_helpers.py -q` | Passed | 3 passed. |
| `python -m pytest tests/test_cli_low_risk_command_contracts.py -q` | Passed | 6 passed. |
| `python run.py --help` | Passed | Command inventory remains 51. |
| `python run.py status` | Passed | Critical text markers present. |
| `python run.py deps` | Passed | Critical text markers present. |
| `python run.py compile-source` | Passed | Critical text markers present. |
| `python run.py repo-assist-eval` | Passed | `23/23`, `uses_web: false`, `uses_model_backend: false`. |
| `python run.py hidden-eval-seed-run` | Passed | 17 attempted, 13 passed, 4 blocked, `private_leakage: false`, `quality_claim: none`. |
| `python run.py trajectory-quality-audit` | Passed | `scanned: 1445`, `sft_positive: 0`, `rejected_positive: 1445`. |
| `python -m pytest tests -q` | Passed | 254 passed, 1 skipped. |
| `python -m compileall -q .` | Passed | No output. |
| `git diff --check` | Passed | No whitespace errors; Git reported an LF/CRLF warning for `run.py`. |
| `git status --short` | Passed | Only expected source/test/report artifacts are dirty or untracked. |
