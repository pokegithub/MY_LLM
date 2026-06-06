# Low-Risk CLI Wrapper Extraction v1

## 1. Executive Verdict

Status/deps/compile-source wrappers extracted: yes.

This package moves only the low-risk command wrapper logic for `status`, `deps`, and `compile-source` into `cli/commands/`. `run.py` still owns parser construction, runtime config loading, dependency gating, and the dispatch table. Parser choices and dispatch keys are unchanged.

Runtime behavior changed: no intended behavior change. JSON contracts and critical text markers remain locked by tests.

Current truth:

- Command inventory: still 51.
- Parser choices changed: no.
- Dispatch keys changed: no.
- JSON schemas changed: no.
- Text markers changed: no.
- Cloud used: no.
- Training started: no.
- Phase B started: no.
- Model quality claimed: no.
- SFT-positive remains 0.

## 2. Wrappers Moved

| Command | Old Location | New Location | Notes |
| --- | --- | --- | --- |
| `status` | `run.py` | `cli/commands/status.py` | `run.py` dispatch calls `cli_show_status(OUTPUT_JSON, _emit_json)`. |
| `deps` | `run.py` | `cli/commands/status.py` | Dependency summary behavior and JSON contract preserved. |
| `compile-source` | `run.py` | `cli/commands/compile_source.py` | Tracked-source-only scope and skipped roots preserved. |

## 3. Helpers Moved

| Helper | New Location | Reason |
| --- | --- | --- |
| `STATUS_TARGETS` | `cli/commands/status.py` | Status wrapper data. |
| `collect_status_checks` | `cli/commands/status.py` | Status report builder support. |
| `collect_dependency_checks` | `cli/commands/status.py` | Dependency report builder support. |
| `collect_backend_dependency_checks` | `cli/commands/status.py` | Optional backend dependency support. Re-exported from `run.py` for existing tests. |
| `_print_dependency_summary` | `cli/commands/status.py` | Text output helper for status/deps. |
| `_print_backend_dependency_summary` | `cli/commands/status.py` | Text output helper for optional backend deps. |
| `_next_status_action` | `cli/commands/status.py` | Status next-action truthfulness support. |
| `build_status_report` | `cli/commands/status.py` | Status JSON payload builder. |
| compile-source skipped root list | `cli/commands/compile_source.py` | Source compile report support. |

## 4. What Did Not Move

- No parser construction moved.
- No command choices changed.
- No dispatch keys changed.
- No high-risk command moved.
- No retrieval/repo-assist command moved.
- No hidden-eval command moved.
- No backend/candidate command moved.
- No training command moved.
- No verifier, trajectory-quality, source-policy, backend normalization, cloud, or data-governance behavior changed.

## 5. Contract Protection

The extraction is protected by:

- `tests/test_cli_command_inventory.py`
- `tests/test_cli_low_risk_command_contracts.py`
- `tests/test_cli_output_helpers.py`

These tests confirm command inventory stability, parser/help/dispatch alignment, JSON contract stability, and critical text marker stability.

## 6. Verification

Verification commands for this package:

- `python -m pytest tests/test_cli_command_inventory.py -q`
- `python -m pytest tests/test_cli_low_risk_command_contracts.py -q`
- `python -m pytest tests/test_cli_output_helpers.py -q`
- `python run.py --help`
- `python run.py --json status`
- `python run.py --json deps`
- `python run.py --json compile-source`
- `python run.py status`
- `python run.py deps`
- `python run.py compile-source`
- `python run.py repo-assist-eval`
- `python run.py hidden-eval-seed-run`
- `python run.py trajectory-quality-audit`
- `python -m pytest tests -q`
- `python -m compileall -q .`
- `git diff --check`
- `git status --short`

Results:

| Command | Result | Notes |
| --- | --- | --- |
| `python -m pytest tests/test_cli_command_inventory.py -q` | Passed | 8 passed; command inventory remains 51. |
| `python -m pytest tests/test_cli_low_risk_command_contracts.py -q` | Passed | 6 passed; JSON contracts and text markers remain stable. |
| `python -m pytest tests/test_cli_output_helpers.py -q` | Passed | 3 passed. |
| `python run.py --help` | Passed | Parser/help surface stable. |
| `python run.py --json status` | Passed | Schema remains `pipeline_status_v2`; `training_quality_claim: none`. |
| `python run.py --json deps` | Passed | Optional backend dependency environment remains separate. |
| `python run.py --json compile-source` | Passed | Schema remains `source_compile_report_v1`; `behavior_changed: false`. |
| `python run.py status` | Passed | Critical text markers present. |
| `python run.py deps` | Passed | Critical text markers present. |
| `python run.py compile-source` | Passed | Critical text markers present. |
| `python run.py repo-assist-eval` | Passed | `23/23`, `uses_web: false`, `uses_model_backend: false`. |
| `python run.py hidden-eval-seed-run` | Passed | 17 attempted, 13 passed, 4 blocked, `private_leakage: false`, `quality_claim: none`. |
| `python run.py trajectory-quality-audit` | Passed | `scanned: 1470`, `sft_positive: 0`, `rejected_positive: 1470`. |
| `python -m pytest tests -q` | Passed | 254 passed, 1 skipped. |
| `python -m compileall -q .` | Passed | No output. |
| `git diff --check` | Passed | No whitespace errors; Git reported an LF/CRLF warning for `run.py`. |
| `git status --short` | Passed | `run.py`, `cli/commands/`, and this report pair are dirty/untracked. |

## 7. Next Recommended Package

Next package: low-risk wrapper extraction follow-up audit or targeted import-boundary cleanup. Do not move retrieval, hidden-eval, backend, trajectory-quality, or training commands until their wrapper contract tests are equally strong.
