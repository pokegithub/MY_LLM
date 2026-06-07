# Hardware / GPU Validate Extraction Readiness v1

## 1. Executive Verdict

`hardware-validate` and `gpu-fit-validate` are not ready for extraction yet.

They are structurally extractable, but extraction should wait until command-level JSON/text/report-path contracts are added. These commands are more hardware-sensitive than `deployment-info` / `hw-profile`: both import `train`, both write reports, and `gpu-fit-validate` can run CUDA fit attempts and checkpoint/resume evidence when CUDA is available.

- `hardware-validate` ready: no
- `gpu-fit-validate` ready: no
- `hardware-validate` risk rating: medium
- `gpu-fit-validate` risk rating: high
- Recommended target module after tests: `cli/commands/hardware.py`
- Recommended strategy: add contract tests first, then extract `hardware-validate` before `gpu-fit-validate`

No handlers were moved in this package.

## 2. Current Command Behavior Summary

| Command | Handler | JSON Schema | Text Marker | Report Path | Executed During Audit |
| --- | --- | --- | --- | --- | --- |
| `hardware-validate` | `run.py::run_hardware_validate` | `hardware_readiness_v1` from `train.run_hardware_readiness_validation()` | `HARDWARE VALIDATION` | `REPORT_PATH` or `DEFAULT_HARDWARE_REPORT_PATH` (`./run_artifacts/hardware_readiness_report.json`) | no |
| `gpu-fit-validate` | `run.py::run_gpu_fit_validate` | `gpu_constrained_fit_validation_v1` from `train.run_gpu_constrained_fit_validation()` | `GPU CONSTRAINED-FIT VALIDATION` | `REPORT_PATH` or `DEFAULT_GPU_FIT_REPORT_PATH` (`./run_artifacts/gpu_constrained_fit_report.json`) | no |

Both commands:

- import `train` lazily inside the handler
- call a validation helper in `train.py`
- write a JSON report via `_write_json_report`
- add `report_path` to the report payload
- honor `OUTPUT_JSON`
- exit nonzero if `report["ok"]` is false
- make no model-quality claim

## 3. Implementation Dependency Map

### `hardware-validate`

- Handler: `run_hardware_validate`
- Output modes: JSON via `_emit_json`; text via `print_banner` and `print`
- JSON schema: `hardware_readiness_v1`
- Helper dependencies in `run.py`: `_write_json_report`, `_emit_json`, `print_banner`
- Global dependencies: `OUTPUT_JSON`, `REPORT_PATH`, `DEFAULT_HARDWARE_REPORT_PATH`
- External import: lazy `import train`
- Main helper: `train.run_hardware_readiness_validation()`
- Hardware dependencies: `torch`, CUDA availability, `nvidia-smi`, tiny CPU/CUDA device step validation
- Report writing: always writes report before output
- Side effects: report file under `run_artifacts`, small tensor/model validation through `train`
- Exit behavior: exits `1` if report `ok` is false

### `gpu-fit-validate`

- Handler: `run_gpu_fit_validate`
- Output modes: JSON via `_emit_json`; text via `print_banner` and `print`
- JSON schema: `gpu_constrained_fit_validation_v1`
- Helper dependencies in `run.py`: `_write_json_report`, `_emit_json`, `print_banner`
- Global dependencies: `OUTPUT_JSON`, `REPORT_PATH`, `DEFAULT_GPU_FIT_REPORT_PATH`
- External import: lazy `import train`
- Main helper: `train.run_gpu_constrained_fit_validation()`
- Hardware dependencies: `torch`, CUDA availability, repo-local interpreter detection, tokenizer artifacts, CUDA fit matrix
- Report writing: always writes report before output
- Side effects: report file under `run_artifacts`; when CUDA is available and interpreter is valid, may create `./run_artifacts/gpu_constrained_fit`, run model/optimizer attempts, write checkpoints, reload checkpoints, and optionally run short real-token validation
- Exit behavior: exits `1` if report `ok` is false

## 4. Coupling / Boundary Audit

| Area | `hardware-validate` Coupling | `gpu-fit-validate` Coupling | Notes |
| --- | --- | --- | --- |
| Training logic | medium | high | Both call `train.py`; `gpu-fit-validate` constructs model/optimizer/checkpoints. |
| Train preflight | low | medium | `gpu-fit-validate` uses default training fit report through preflight. |
| Checkpoint files | none | high | `gpu-fit-validate` writes/reloads checkpoint evidence for fit attempts. |
| Hidden eval | none | none | No hidden target access. |
| Retrieval | none | none | No retrieval/citation path. |
| Backend/parser normalization | none | none | No candidate backend dependency. |
| Trajectory quality | none | none | Does not affect SFT-positive eligibility. |
| Cloud scripts | none | none | No cloud commands or downloads. |
| Model downloads | none | none | No provisioning behavior. |
| Data governance | none | none | Separate command. |
| Hardware profile config | low | low | Indirect train/hardware policy context only. |
| Torch/CUDA | medium | high | `hardware-validate` runs tiny validation; `gpu-fit-validate` can run CUDA matrix. |
| Tokenizer/model construction | medium | high | Both can construct tiny validation model; `gpu-fit-validate` also loads tokenizer artifacts. |
| `run_artifacts` report writing | medium | high | Both write reports; `gpu-fit-validate` can write checkpoint artifacts too. |

## 5. Current Test Coverage

Covered today:

- Command presence through `tests/test_cli_command_inventory.py`
- Parser/dispatch inventory count remains 51
- Function-level hardware evidence checks in `tests/test_hard_evidence.py`
- `run_hardware_readiness_validation()` report schema and no fake training/model-quality claims
- GPU fit classification helper has no fake success state
- GPU fit report structure when CUDA is available, using a temp checkpoint root and `run_short_validation=False`

Missing before extraction:

- `hardware-validate --json` CLI contract
- `hardware-validate` text marker contract
- `hardware-validate` report-path behavior with temporary report path
- `hardware-validate` CLI no-overclaim checks
- `gpu-fit-validate --json` CLI contract or explicit safe skip rule
- `gpu-fit-validate` text marker contract or explicit safe skip rule
- `gpu-fit-validate` report-path behavior with temporary report path and mocked/safe no-CUDA path
- future `cli.commands.hardware` import-safety test
- future module import does not inspect hardware
- explicit no cloud/download/model-loading assertion for the command-contract tests

## 6. Output / Report Side-Effect Analysis

The commands were not executed during this audit.

Reason: this package is a readiness audit, not a contract-test or evidence-refresh package. `hardware-validate` is probably bounded but still performs live hardware validation and writes a report. `gpu-fit-validate` is more sensitive: when CUDA is available and the interpreter is repo-local, it can allocate CUDA memory, run multiple fit attempts, write checkpoint evidence, reload checkpoints, and optionally run short real-token validation.

The next test package should exercise these command surfaces using temporary report paths and mocks/safe skip rules where needed.

## 7. Proposed Target Module Layout

Recommended future module:

`cli/commands/hardware.py`

Recommended future function shapes:

- `run_hardware_validate(output_json: bool, emit_json, write_json_report, report_path=None) -> None`
- `run_gpu_fit_validate(output_json: bool, emit_json, write_json_report, report_path=None) -> None`

Potential pure helpers if tests justify them:

- `print_hardware_validation_report(report: dict, report_path: str) -> None`
- `print_gpu_fit_validation_report(report: dict, report_path: str) -> None`

Import rule for extraction: keep `train` imports lazy inside command functions. Importing `cli.commands.hardware` should not import `train`, `torch`, allocate tensors, inspect CUDA, load tokenizers, write reports, or create `run_artifacts`.

## 8. Risk Rating

| Command | Risk | Reason |
| --- | --- | --- |
| `hardware-validate` | medium | Bounded, but imports `train`, performs live torch/hardware validation, writes a report, and exits nonzero on failed hardware evidence. Missing CLI JSON/text/report-path contracts. |
| `gpu-fit-validate` | high | May allocate CUDA memory, construct models/optimizers, write/reload checkpoints, load tokenizer artifacts, run fit matrix, run short validation, and write report/checkpoint artifacts. Missing CLI contracts and safe skip/mocking strategy. |

## 9. Required Tests Before Extraction

1. `hardware-validate --json` contract test.
2. `hardware-validate` text marker test.
3. `hardware-validate` temporary `--report-path` behavior test.
4. `hardware-validate` no training/Phase B/model-quality overclaim test.
5. `gpu-fit-validate --json` contract test using a safe no-CUDA/mocked path, or an explicit skip rule.
6. `gpu-fit-validate` text marker test using a safe no-CUDA/mocked path, or an explicit skip rule.
7. `gpu-fit-validate` temporary `--report-path` behavior test without writing to default `run_artifacts`.
8. Future `cli.commands.hardware` import-safety test.
9. Future module import does not import `train`/`torch`, inspect hardware, or write files.
10. Command inventory remains 51.

## 10. Extraction Recommendation

Recommended option: add contract tests first, then extract only `hardware-validate` before `gpu-fit-validate`.

Exact next package:

`hardware/gpu validate CLI contract tests`

After that passes, extract `hardware-validate` into `cli/commands/hardware.py`. Keep `gpu-fit-validate` for a later package unless its contract tests and safe execution strategy are clearly stable.

## 11. Artifact Safety Result

Artifact safety commands showed:

- `git diff --cached --name-only`: empty
- `git ls-files run_artifacts data_cache checkpoints sft_checkpoints dpo_checkpoints distill_checkpoints improved_checkpoints quantized eval_results`: empty
- no `run_artifacts/` files staged
- no model files staged
- no cache/checkpoint roots staged
- no protected generated roots tracked unexpectedly

`git diff --name-only` currently reports `run.py` because the accepted previous wrapper extraction remains uncommitted in this workspace.

## 12. Verification Results

Verification was run after this report was created; see final package response for exact command results.

Package invariants:

- Handlers moved: no
- Runtime behavior changed: no
- Command inventory changed: no, still 51
- Cloud used: no
- Training started: no
- Phase B started: no
- Model quality claim: none
- SFT-positive: 0

## 13. Next Recommended Package

`hardware/gpu validate CLI contract tests`

Scope: add command-level contracts and safe report-path tests for `hardware-validate` and safe/mocked or skipped contracts for `gpu-fit-validate`, without extracting handlers yet.
