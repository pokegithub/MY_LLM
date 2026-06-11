# GPU Fit Validate Extraction Contracts v1

## 1. Executive Verdict

- gpu-fit extraction contract tests strengthened: yes
- real `gpu-fit-validate` run: no
- CUDA allocation avoided: yes
- checkpoint artifacts written: no
- mocked JSON contract: passed
- mocked text contract: passed
- `ok:false` exit behavior: covered
- default report path untouched: yes
- checkpoint artifact safety: covered in mocked path
- future import safety requirements: recorded
- extraction allowed next: yes, guarded
- recommended target module: `cli/commands/gpu_fit.py`
- command inventory: still 51
- runtime behavior changed: no
- cloud used: no
- training started: no
- Phase B started: no
- model quality claim: none
- SFT-positive: 0

## 2. Tests Added Or Strengthened

Updated `tests/test_cli_hardware_validate_contracts.py`:

- strengthened mocked JSON contract for `run.run_gpu_fit_validate`
- strengthened mocked text contract for `run.run_gpu_fit_validate`
- added mocked JSON `ok:false` exit behavior coverage
- added explicit default GPU-fit report path snapshot checks
- added explicit `run_artifacts/gpu_constrained_fit` checkpoint/artifact root snapshot checks
- added temp-directory write boundary checks
- added fixture assertions for future `cli.commands.gpu_fit` import-safety requirements

Updated `tests/fixtures/cli_hardware_validate_contracts_v1.json`:

- recorded future module name: `cli.commands.gpu_fit`
- recorded temp report requirement
- recorded `ok:false` exit behavior requirement
- recorded no-default-report-path-write requirement
- recorded no-checkpoint-artifact-creation requirement
- recorded future import-safety requirements:
  - no `train` import at module import time
  - no `torch` import at module import time
  - no CUDA inspection at module import time
  - no file writes at module import time

## 3. Mocked JSON Contract

Result: passed.

The JSON contract test still uses a fake `train.run_gpu_constrained_fit_validation` implementation and calls the existing handler directly. It does not launch `python run.py gpu-fit-validate`, does not call real CUDA code, does not construct models, and does not write checkpoint artifacts.

The test asserts:

- schema `gpu_constrained_fit_validation_v1`
- required JSON keys
- `quality_claim: none`
- `pretraining_readiness_changed: no`
- temp report path is used
- temp report is written
- default GPU-fit report path is untouched
- `run_artifacts/gpu_constrained_fit` is untouched
- writes remain inside the temporary directory

## 4. Mocked Text Contract

Result: passed.

The text contract test still uses a fake `train.run_gpu_constrained_fit_validation` implementation and captures stdout from the handler.

The test asserts:

- text marker `GPU CONSTRAINED-FIT VALIDATION`
- `cuda_available`
- `hardware_class`
- `Fit matrix`
- `report_path`
- `quality_claim      : none`
- temp report path is used
- default GPU-fit report path is untouched
- `run_artifacts/gpu_constrained_fit` is untouched
- no training, Phase B, or model-quality overclaim appears

## 5. `ok:false` Exit Behavior Contract

Result: covered.

The new mocked failure test returns `ok: false` from the fake GPU-fit report and asserts:

- `run.run_gpu_fit_validate()` raises `SystemExit`
- exit code is `1`
- JSON is emitted before exit
- temp report is written before exit
- temp report path is preserved
- default GPU-fit report path is untouched
- `run_artifacts/gpu_constrained_fit` is untouched
- `quality_claim` remains `none`
- `pretraining_readiness_changed` remains `no`

## 6. Default Report Path Safety

Result: covered.

Tests snapshot `run_artifacts/gpu_constrained_fit_report.json` before and after mocked GPU-fit handler calls. When a temp `REPORT_PATH` is supplied, the default report path is not created or modified.

## 7. Checkpoint Artifact Safety

Result: covered for mocked handler path.

Tests snapshot `run_artifacts/gpu_constrained_fit` before and after mocked GPU-fit handler calls. The mocked path does not create or modify the real GPU-fit artifact root.

This does not prove the real CUDA command is artifact-free. The real command is explicitly not run in unit tests.

## 8. Future Import-Safety Requirements

Recorded in fixture for the future extraction package:

- future module: `cli.commands.gpu_fit`
- importing the future module must be quiet
- importing the future module must not import `train`
- importing the future module must not import `torch`
- importing the future module must not inspect CUDA
- importing the future module must not write files
- future module must expose `run_gpu_fit_validate`
- real CUDA subprocess execution remains disallowed in unit tests

No production `cli/commands/gpu_fit.py` module was created in this package.

## 9. What Is Still Not Allowed

- no real `python run.py gpu-fit-validate`
- no CUDA allocation
- no model construction
- no optimizer construction
- no checkpoint writes
- no short real-token validation
- no cloud execution
- no model downloads
- no training
- no SFT/DPO/RLVR/distillation
- no Phase B
- no model-quality claims

## 10. Extraction Readiness After This Package

Extraction allowed next: yes, guarded.

The next package may move only the wrapper into `cli/commands/gpu_fit.py` if it also adds and passes the future module import-safety test. The extraction package must keep the same mocked JSON/text/`ok:false`/report-path/artifact-safety contracts and must still avoid real CUDA execution.

## 11. Verification Results

Targeted:

- `python -m pytest tests/test_cli_hardware_validate_contracts.py -q`: 6 passed

Full verification results are recorded in the final package response.

## 12. Next Recommended Package

Next package: `extract gpu-fit-validate wrapper only`.

Constraints for next package:

- target module: `cli/commands/gpu_fit.py`
- move only the CLI wrapper
- keep `train` import lazy
- add import-safety test for the new module
- do not run real `gpu-fit-validate`
- do not allocate CUDA
- do not change schema/text/report-path/exit behavior
