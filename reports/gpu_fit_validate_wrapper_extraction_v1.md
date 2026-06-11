# GPU Fit Validate Wrapper Extraction v1

## 1. Executive Verdict

- gpu-fit-validate wrapper extracted: yes
- target module: `cli/commands/gpu_fit.py`
- import safe: yes
- `train` imported on module import: no
- `torch` imported on module import: no
- CUDA inspected on module import: no
- files written on module import: no
- mocked JSON contract: passed
- mocked text contract: passed
- `ok:false` exit behavior: preserved
- default report path untouched: yes
- checkpoint artifacts written: no
- real `gpu-fit-validate` run: no
- command inventory: still 51
- runtime behavior changed: no
- cloud used: no
- training started: no
- Phase B started: no
- model quality claim: none
- SFT-positive: 0

## 2. Wrapper Moved

Moved only the `gpu-fit-validate` CLI wrapper from `run.py` into `cli/commands/gpu_fit.py`.

The extracted wrapper preserves:

- lazy `train` import inside `run_gpu_fit_validate`
- call to `train.run_gpu_constrained_fit_validation()`
- supplied JSON emitter callback
- supplied JSON report writer callback
- explicit `REPORT_PATH` before default path fallback
- JSON output behavior
- text output marker `GPU CONSTRAINED-FIT VALIDATION`
- fit matrix, short validation, default fit, report path, and caveat text output
- nonzero exit on `report["ok"] == false`
- `quality_claim      : none`

## 3. Import-Safety Result

`tests/test_cli_gpu_fit_command_imports.py` verifies importing `cli.commands.gpu_fit`:

- succeeds
- prints nothing to stdout/stderr
- does not import `train`
- does not import `torch`
- does not change the default GPU-fit report path
- does not change `run_artifacts/gpu_constrained_fit`
- exposes `run_gpu_fit_validate`

## 4. Mocked Contract Preservation

`tests/test_cli_hardware_validate_contracts.py` now calls the extracted wrapper directly with a fake `train` module and local callbacks.

Covered:

- mocked JSON contract
- mocked text contract
- mocked `ok:false` exit behavior
- temp report path preservation
- default report path safety
- checkpoint artifact root safety
- no training, Phase B, or model-quality overclaim

Real CUDA `gpu-fit-validate` was not run.

## 5. What Was Not Run

- no `python run.py gpu-fit-validate`
- no real CUDA GPU-fit validation
- no model construction
- no optimizer construction
- no checkpoint evidence write
- no short real-token validation
- no cloud command
- no training command
- no SFT/DPO/RLVR/distillation
- no Phase B

## 6. Verification Results

Targeted checks before full verification:

- `python -m pytest tests/test_cli_hardware_validate_contracts.py -q`: 6 passed
- `python -m pytest tests/test_cli_gpu_fit_command_imports.py -q`: 1 passed

Full verification results are recorded in the final package response.

## 7. Next Recommended Package

Next package: `gpu-fit extraction audit`, checking for stale duplicate wrapper code in `run.py`, import-boundary safety, line-count reduction, and contract stability.
