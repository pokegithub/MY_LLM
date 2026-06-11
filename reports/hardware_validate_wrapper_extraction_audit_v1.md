# Hardware Validate Wrapper Extraction Audit v1

## 1. Executive Verdict

- hardware-validate extraction audit created: yes
- hardware-validate extracted: yes
- target module: `cli/commands/hardware.py`
- stale duplicate implementation in `run.py`: no
- `gpu-fit-validate` still in `run.py`: yes
- import boundary safe: yes
- lazy `train` import preserved: yes
- lazy `torch` import preserved: yes
- hardware inspected on import: no
- reports written on import: no
- command inventory: still 51
- JSON schema changed: no
- text markers changed: no
- report-path behavior changed: no
- exit behavior changed: no
- runtime behavior changed: no
- real `gpu-fit-validate` run: no
- cloud used: no
- training started: no
- Phase B started: no
- model quality claim: none
- SFT-positive: 0

## 2. Stale Duplicate Check

Static search found no old full `def run_hardware_validate` implementation in `run.py`.

Evidence:
- `run.py` imports `cli_run_hardware_validate` from `cli.commands.hardware`.
- `run.py` dispatches `hardware-validate` through the extracted wrapper.
- `run.py` no longer contains the direct `HARDWARE VALIDATION` output block.
- `run.py` no longer calls `train.run_hardware_readiness_validation()` for `hardware-validate`.
- `run_gpu_fit_validate` remains defined in `run.py`.

## 3. Import Boundary Check

`cli/commands/hardware.py` has a one-way boundary:

- imports `cli.output.print_banner`
- does not import `run.py`
- does not import `train` at module import time
- does not import `torch` at module import time
- imports `train` only inside `run_hardware_validate`
- exposes `run_hardware_validate`
- does not expose `run_gpu_fit_validate`

Import-safety test result:

- `tests/test_cli_hardware_command_imports.py`: 1 passed

## 4. Contract Preservation

`tests/test_cli_hardware_validate_contracts.py` passed and preserves:

- `hardware-validate --json --report-path <tmp>` behavior
- temporary report-path usage
- no default report-path write in targeted tests
- schema `hardware_readiness_v1`
- text marker `HARDWARE VALIDATION`
- exit behavior tied to `report["ok"]`
- no training, Phase B, or model-quality overclaim
- no real CUDA `gpu-fit-validate` execution

## 5. Line Counts And Responsibility

- previous known `run.py` count after deployment/hw-profile audit: 2578
- current `run.py` count: 2537
- reduction: 41 lines
- `cli/commands/hardware.py`: 74 lines
- `cli/commands/deployment.py`: 137 lines
- `cli/commands/status.py`: 224 lines
- `cli/commands/compile_source.py`: 90 lines

Remaining `run.py` responsibilities:

- parser construction
- runtime config application
- dependency gating
- dispatch table
- remaining command wrappers
- `gpu-fit-validate`
- high-risk training/eval/data command wrappers
- retrieval/repo-assist command wrappers
- hidden-eval command wrappers
- backend/candidate command wrappers
- trajectory-quality command wrappers

## 6. Artifact Safety

Artifact safety checks:

- `git diff --cached --name-only`: no staged files
- `git ls-files run_artifacts data_cache checkpoints sft_checkpoints dpo_checkpoints distill_checkpoints improved_checkpoints quantized eval_results`: no tracked protected artifact roots reported
- no `run_artifacts`, model files, cache roots, checkpoints, or generated roots staged by this package

## 7. Verification Results

Focused checks:

- `python -m pytest tests/test_cli_command_inventory.py -q`: 8 passed
- `python -m pytest tests/test_cli_hardware_validate_contracts.py -q`: 5 passed
- `python -m pytest tests/test_cli_hardware_command_imports.py -q`: 1 passed
- `python -m pytest tests/test_cli_low_risk_command_contracts.py -q`: 6 passed
- `python -m pytest tests/test_cli_informational_command_contracts.py -q`: 4 passed
- `python -m pytest tests/test_cli_deployment_command_imports.py -q`: 1 passed
- `python -m pytest tests/test_cli_output_helpers.py -q`: 3 passed

Command checks:

- `python run.py --help`: passed, command inventory includes 51 choices
- `python run.py status`: passed
- `python run.py deps`: passed
- `python run.py compile-source`: passed, 117 tracked Python sources scanned
- `python run.py deployment-info`: passed
- `python run.py hw-profile`: passed
- `python run.py repo-assist-eval`: 23 passed, 0 failed
- `python run.py hidden-eval-seed-run`: 13 passed, 1 failed, 3 blocked, private leakage false
- `python run.py trajectory-quality-audit`: scanned 1726, SFT-positive 0
- `python -m pytest tests -q`: 265 passed, 1 skipped
- `python -m compileall -q .`: passed

## 8. Remaining Risks

- `gpu-fit-validate` remains in `run.py` by design and should not be extracted without a separate readiness audit and contract coverage.
- The hidden eval run still has non-passed items; this does not affect the wrapper audit, but it remains real evidence that no model-quality claim is justified.
- `run.py` remains large and high-risk. Further extraction should proceed only command family by command family with contract tests first.

## 9. Next Recommended Package

Next safe package: `gpu-fit-validate extraction readiness audit`, with no extraction and no real CUDA execution. The audit should identify required contract tests, import-boundary risks, and report-path/exit-code expectations before any wrapper movement.
