# GPU Fit Validate Extraction Readiness v1

## 1. Executive Verdict

- gpu-fit-validate extraction readiness audit created: yes
- gpu-fit-validate ready to extract: no
- risk rating: high
- recommended target module: `cli/commands/gpu_fit.py`
- recommended strategy: add extraction-specific mocked handler and import-safety tests first, then move only the CLI wrapper
- real `gpu-fit-validate` run: no
- CUDA allocation avoided: yes
- checkpoint artifacts written: no
- command inventory: still 51
- runtime behavior changed: no
- cloud used: no
- training started: no
- Phase B started: no
- model quality claim: none
- SFT-positive: 0

## 2. Current Implementation Summary

`run_gpu_fit_validate` currently remains in `run.py`. Its wrapper behavior is:

- lazily imports `train`
- calls `train.run_gpu_constrained_fit_validation()`
- writes a JSON report through `_write_json_report`
- uses explicit `REPORT_PATH` when provided, otherwise `DEFAULT_GPU_FIT_REPORT_PATH`
- appends `report_path` into the emitted payload
- emits JSON via `_emit_json` when `--json` is active
- exits nonzero when `report["ok"]` is false
- prints text marker `GPU CONSTRAINED-FIT VALIDATION`
- prints interpreter, CUDA, VRAM, dtype, fit matrix, short validation, default fit, readiness-change, report path, caveats, and `quality_claim      : none`

The wrapper is behaviorally small, but the function it invokes is high-risk.

## 3. Side-Effect And Risk Audit

| Side Effect | Classification | Evidence |
| --- | --- | --- |
| Allocate CUDA memory | yes | `train._run_gpu_fit_attempt` uses CUDA tensors, model `.to(device)`, optimizer step, and CUDA memory stats. |
| Construct model objects | yes | `LLM(cfg).to(device).train()` in `_run_gpu_fit_attempt`. |
| Construct optimizer objects | yes | `build_adamw(...)` in `_run_gpu_fit_attempt`. |
| Run fit matrix attempts | yes | `run_gpu_constrained_fit_validation` iterates matrix candidates. |
| Write checkpoint evidence | yes when real CUDA path runs | `_run_gpu_fit_attempt` calls `save_ckpt(... tag="_gpu_fit")`. |
| Reload checkpoints | yes when real CUDA path runs | `_run_gpu_fit_attempt` calls `load_ckpt(...)`. |
| Run short real-token validation | possible | If any matrix candidate fits and `run_short_validation=True`, it calls `run_short_real_data_validation(... force_device_type="cuda")`. |
| Touch tokenizer artifacts | yes | `BPETokenizer().load(tokenizer_path)`. |
| Write under `run_artifacts/gpu_constrained_fit` | yes when CUDA path passes interpreter/CUDA gates | Default `checkpoint_root` is `./run_artifacts/gpu_constrained_fit`. |
| Write default GPU-fit report path | yes if command is executed without `--report-path` | Wrapper writes `DEFAULT_GPU_FIT_REPORT_PATH`. |
| Exit nonzero based on `report["ok"]` | yes | Wrapper exits when `report["ok"]` is false. |
| Create machine-specific output | yes | CUDA availability, VRAM, fit matrix, memory peaks, and runtime depend on host hardware. |

Readiness risk rating: high.

## 4. Coupling Audit

| Area | Coupling | Notes |
| --- | --- | --- |
| `train.py` | high | The command is a thin wrapper around `train.run_gpu_constrained_fit_validation`. |
| model construction | high | Real path constructs `LLM` instances. |
| optimizer/checkpoint code | high | Real path builds optimizer, saves and reloads checkpoints. |
| tokenizer artifacts | medium | Tokenizer load is required after CUDA/interpreter gates. |
| short validation | high | Short validation can run after a fitting matrix candidate. |
| train preflight | medium | Default training fit calls preflight reporting. |
| hardware validation | low | Related domain, separate command path. |
| data cache | medium | Short validation can use token cache. |
| checkpoint roots | high | Real path writes checkpoint evidence. |
| cloud scripts | low | No direct coupling found. |
| hidden eval | low | No direct coupling found. |
| retrieval/repo-assist | low | No direct coupling found. |
| trajectory-quality | low | No direct coupling found. |
| backend/candidate logic | low | No direct coupling found. |

## 5. Existing Contract Coverage

Existing coverage from `tests/test_cli_hardware_validate_contracts.py` and `tests/fixtures/cli_hardware_validate_contracts_v1.json`:

- command inventory presence: covered
- mocked JSON contract: covered
- mocked text contract: covered
- mocked temp report path behavior: covered
- no-overclaim behavior: covered
- direct real CUDA execution disallowed: covered by fixture/contract assertion
- safe no-CUDA path through real subprocess: not covered for `gpu-fit-validate`
- exit behavior for `ok: false`: not sufficiently covered for `gpu-fit-validate`
- import safety of a future module: not yet covered
- no checkpoint artifact creation in extracted mocked path: not yet covered
- no default report-path modification in mocked GPU-fit path: partially implied by temp path use, but should be explicit before extraction
- no training/Phase B/model-quality claim: covered

Current tests are enough to prevent accidental real CUDA execution in unit tests. They are not enough to extract safely without adding module import-safety and failure/side-effect tests.

## 6. Future Target Module Decision

Recommended target module: `cli/commands/gpu_fit.py`.

Reason:

- `gpu-fit-validate` is much heavier than `hardware-validate`.
- Its underlying function can allocate CUDA memory, construct models/optimizers, write checkpoints, reload checkpoints, and run short validation.
- A separate module gives stricter import-safety boundaries and lets tests assert that importing GPU-fit command code does not import `train`, `torch`, inspect CUDA, or create artifacts.

Do not add it to `cli/commands/hardware.py` until the extraction-specific tests exist. Keeping the high-risk command separate will make future review easier.

## 7. Extraction Readiness Verdict

Verdict: not ready; add more tests first.

Reasoning:

- The wrapper itself is small, but the invoked function is high side-effect and machine-specific.
- Current GPU-fit tests use mocked handler execution only.
- No future-module import-safety test exists yet.
- No explicit `ok: false` exit preservation test exists for GPU-fit extraction.
- No explicit no-default-report-path-write test exists for the mocked GPU-fit path.
- No explicit no-checkpoint-artifact-creation test exists for the extracted mocked path.

Future extraction is allowed only after these tests are added and green.

## 8. Required Tests Before Extraction

1. Future module import-safety test for `cli.commands.gpu_fit`.
2. No `train` import at module import time.
3. No `torch` import at module import time.
4. No CUDA inspection at module import time.
5. No files written at module import time.
6. Handler-level mocked JSON contract after extraction.
7. Handler-level mocked text contract after extraction.
8. Temp report-path preservation test.
9. Exit behavior preservation test for `ok: false`.
10. No default report-path write test.
11. No checkpoint artifact creation test in mocked path.
12. Direct real CUDA execution remains disallowed in unit tests.
13. Command inventory remains 51.
14. No training/Phase B/model-quality overclaim test.

## 9. Artifact Safety

- `git diff --cached --name-only`: no staged files at audit time
- protected artifact roots tracked by `git ls-files`: none reported
- real `gpu-fit-validate` was not run
- CUDA memory allocation was avoided
- checkpoint artifacts were not written
- no `run_artifacts`, model files, cache roots, checkpoints, or generated roots were staged by this package

## 10. Verification Results

Precheck/focused tests:

- `python -m pytest tests/test_cli_command_inventory.py -q`: 8 passed
- `python -m pytest tests/test_cli_hardware_validate_contracts.py -q`: 5 passed
- `python -m pytest tests/test_cli_hardware_command_imports.py -q`: 1 passed
- `python -m pytest tests/test_cli_low_risk_command_contracts.py -q`: 6 passed
- `python -m pytest tests/test_cli_informational_command_contracts.py -q`: 4 passed
- `python -m pytest tests/test_cli_deployment_command_imports.py -q`: 1 passed
- `python -m pytest tests/test_cli_output_helpers.py -q`: 3 passed

Full verification results are recorded in the final package response.

## 11. Next Recommended Package

Next package: `gpu-fit-validate extraction contract tests`, still with no extraction and no real CUDA execution.

That package should add the missing mocked/import-safety tests listed above. Only after those pass should a later package move the wrapper into `cli/commands/gpu_fit.py`.
