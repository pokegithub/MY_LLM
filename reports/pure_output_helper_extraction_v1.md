# Pure Output Helper Extraction v1

## 1. Executive Verdict

`cli/output.py` added: yes.

Pure output helpers extracted: yes. This package adds a tiny CLI output helper module and replaces repeated banner/boolean display formatting in `run.py`. Command handlers were not moved. Parser construction, dispatch mapping, config loading, backend logic, retrieval logic, hidden-eval logic, trajectory-quality logic, and training logic remain in place.

Runtime behavior changed: no intended behavior change. This is a behavior-preserving output-helper extraction.

Current truth:

- Command inventory changed: no, still 51 commands.
- Command handlers moved: no.
- Cloud used: no.
- Training started: no.
- Phase B started: no.
- Model quality claimed: no.
- SFT-positive remains 0.

## 2. Helpers Extracted

| Helper | Path | Purpose | Side Effects |
| --- | --- | --- | --- |
| `print_banner(title)` | `cli/output.py` | Prints the standard three-line `run.py` section banner. | Printing only. |
| `format_bool(value)` | `cli/output.py` | Preserves the existing lowercase boolean-like CLI text convention. | None. |
| `safe_text(value)` | `cli/output.py` | Converts printable values while rendering `None` as `none`. | None. |

## 3. What Was Not Moved

- No command handlers were moved.
- No parser choices were changed.
- No dispatch keys were changed.
- No JSON report schemas were changed.
- No report paths were changed.
- No verifier, backend, retrieval, hidden-eval, trajectory-quality, cloud, or training behavior was changed.

## 4. Safety Rationale

The extracted helpers are pure/simple output helpers. They do not call model, backend, retrieval, hidden-eval, training, cloud, filesystem mutation, or report-writing logic. The command inventory snapshot tests lock the command surface after the edit.

## 5. Verification

Verification commands for this pass:

- `python -m pytest tests/test_cli_command_inventory.py -q`
- `python run.py --help`
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
| `python -m pytest tests/test_cli_command_inventory.py -q` | Passed | 8 passed; parser/help/dispatch snapshot still matches. |
| `python -m pytest tests/test_cli_output_helpers.py -q` | Passed | 3 passed. |
| `python run.py --help` | Passed | Command inventory remains 51. |
| `python run.py status` | Passed | No training readiness claim. |
| `python run.py deps` | Passed | Core deps 7/0; optional backend deps 2/0. |
| `python run.py compile-source` | Passed | `scanned: 103`, tracked-source-only scope. |
| `python run.py repo-assist-eval` | Passed | `23/23`, `uses_web: false`, `uses_model_backend: false`. |
| `python run.py hidden-eval-seed-run` | Passed | 17 attempted, 13 passed, 4 blocked, `private_leakage: false`, `quality_claim: none`. |
| `python run.py trajectory-quality-audit` | Passed | `scanned: 1420`, `sft_positive: 0`, `rejected_positive: 1420`. |
| `python -m pytest tests -q` | Passed | 248 passed, 1 skipped. |
| `python -m compileall -q .` | Passed | No output. |
| `git diff --check` | Passed | No whitespace errors; Git reported an LF/CRLF warning for `run.py`. |
| `git status --short` | Passed | Source diff is `run.py`; new helper/test/report files are untracked. |

## 6. Next Recommended Package

Next package: output/helper extraction follow-up only if needed, otherwise low-risk `status`/`deps`/`compile-source` wrapper extraction planning.

Do not move wrappers until the command inventory snapshot remains green.
