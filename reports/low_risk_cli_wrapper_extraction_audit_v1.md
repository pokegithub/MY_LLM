# Low-Risk CLI Wrapper Extraction Audit v1

## 1. Executive Verdict

Low-risk wrapper extraction audit created: yes.

The `status`, `deps`, and `compile-source` extraction is clean. No stale duplicate full implementations remain in `run.py`. The import boundary is one-way: `run.py` imports `cli.commands.*`, while `cli/output.py`, `cli/commands/status.py`, and `cli/commands/compile_source.py` do not import `run.py`.

Runtime behavior changed in this audit package: no. No additional command extraction was performed.

Current truth:

- Stale duplicate implementation in `run.py`: no.
- Import boundary safe: yes.
- Circular import risk found: no.
- Command inventory: still 51.
- Low-risk contracts: passed.
- Cloud used: no.
- Training started: no.
- Phase B started: no.
- Model quality claimed: no.
- SFT-positive remains 0.

## 2. Stale Duplicate Check

Searches checked that these definitions no longer exist in `run.py`:

- `STATUS_TARGETS`
- `collect_status_checks`
- `collect_dependency_checks`
- `_print_dependency_summary`
- `_print_backend_dependency_summary`
- `_next_status_action`
- `build_status_report`
- old full `show_status`
- old full `run_deps`
- old full `run_compile_source`

Result: no stale full duplicate implementation found in `run.py`.

Intentional compatibility boundary:

- `run.py` re-exports `collect_backend_dependency_checks` by importing it from `cli.commands.status` because an existing backend test imports that helper from `run.py`.
- This is compatibility surface only; the implementation lives in `cli/commands/status.py`.

Dispatch result:

- `deps` dispatches to `cli_run_deps(OUTPUT_JSON, _emit_json)`.
- `compile-source` dispatches to `cli_run_compile_source(OUTPUT_JSON, _emit_json)`.
- `status` dispatches to `cli_show_status(OUTPUT_JSON, _emit_json)`.

## 3. Import-Boundary Check

Import checks:

- `cli.output`: import succeeded.
- `cli.commands.status`: import succeeded.
- `cli.commands.compile_source`: import succeeded.

Boundary checks:

- `cli/output.py` does not import `run.py`.
- `cli/commands/status.py` does not import `run.py`.
- `cli/commands/compile_source.py` does not import `run.py`.
- `run.py` imports the new modules one-way.
- No command handler executes during module import.

Circular import risk: no current evidence of circular import risk.

## 4. `run.py` Responsibility / Line-Count Summary

Current line counts:

| File | Lines |
| --- | ---: |
| `run.py` | 2626 |
| `cli/commands/status.py` | 224 |
| `cli/commands/compile_source.py` | 90 |
| `cli/output.py` | 23 |

Previous known `run.py` line count from the professional repo audit: 2704.

Approximate `run.py` reduction: 78 lines.

Remaining `run.py` responsibilities:

- parser construction
- runtime config application
- dependency gating
- dispatch table
- high-risk command wrappers
- retrieval/repo-assist wrappers
- hidden-eval wrappers
- backend/candidate wrappers
- training/eval/data/governance wrappers

This is acceptable for this stage. More movement should wait for equivalent contract tests around the next command group.

## 5. Contract-Test Summary

Precheck tests:

- `tests/test_cli_command_inventory.py`: passed.
- `tests/test_cli_low_risk_command_contracts.py`: passed.
- `tests/test_cli_output_helpers.py`: passed.

Behavior lock status:

- Parser choices unchanged.
- Dispatch keys unchanged.
- Command inventory remains 51.
- `status`, `deps`, and `compile-source` JSON contracts remain locked.
- Critical text markers remain locked.

## 6. Artifact Safety Result

Artifact safety commands:

- `git diff --name-only`: `run.py`
- `git diff --cached --name-only`: empty
- `git ls-files run_artifacts data_cache checkpoints sft_checkpoints dpo_checkpoints distill_checkpoints improved_checkpoints quantized eval_results`: empty

Result:

- No `run_artifacts/` files staged.
- No generated model/cache/checkpoint roots staged.
- No protected generated roots tracked unexpectedly by this audit.

## 7. Remaining Risks

- `run.py` is still large and owns many command wrappers.
- Next extraction group should not be retrieval, hidden-eval, backend, trajectory-quality, or training until matching contract tests exist.
- Training commands must remain last.
- Hidden/private and citation/source-policy behavior should not be moved casually.

## 8. Next Recommended Package

Next package: low-risk wrapper extraction follow-up audit is complete; the next safe implementation package should add contract tests for the next candidate command group before any movement.

Recommended candidate:

- Add contract tests for `deployment-info` and `hw-profile`, or
- Add an extraction plan for repo-assist wrappers without moving them yet.

Do not move retrieval, hidden-eval, backend, trajectory-quality, or training commands next.

## 9. Verification

Verification commands for this audit package:

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
| `python -m pytest tests/test_cli_low_risk_command_contracts.py -q` | Passed | 6 passed; low-risk JSON/text contracts stable. |
| `python -m pytest tests/test_cli_output_helpers.py -q` | Passed | 3 passed. |
| `python run.py --help` | Passed | Parser/help surface stable. |
| `python run.py --json status` | Passed | `pipeline_status_v2`, `training_quality_claim: none`. |
| `python run.py --json deps` | Passed | Optional backend dependency environment remains separate. |
| `python run.py --json compile-source` | Passed | `source_compile_report_v1`, `behavior_changed: false`. |
| `python run.py status` | Passed | Critical text markers present. |
| `python run.py deps` | Passed | Critical text markers present. |
| `python run.py compile-source` | Passed | Critical text markers present. |
| `python run.py repo-assist-eval` | Passed | `23/23`, `uses_web: false`, `uses_model_backend: false`. |
| `python run.py hidden-eval-seed-run` | Passed | 17 attempted, 13 passed, 4 blocked, `private_leakage: false`, `quality_claim: none`. |
| `python run.py trajectory-quality-audit` | Passed | `scanned: 1495`, `sft_positive: 0`, `rejected_positive: 1495`. |
| `python -m pytest tests -q` | Passed | 254 passed, 1 skipped. |
| `python -m compileall -q .` | Passed | No output. |
| `git diff --check` | Passed | No whitespace errors; Git reported an LF/CRLF warning for `run.py`. |
| `git status --short` | Passed | Expected wrapper extraction files and reports only. |
