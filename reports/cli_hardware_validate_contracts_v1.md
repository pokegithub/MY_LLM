# CLI Hardware Validate Contracts v1

## 1. Executive Verdict

Command-level contracts were added for `hardware-validate` and a safe mocked/non-real-CUDA contract was added for `gpu-fit-validate`.

- `hardware-validate` contract: locked
- `hardware-validate` temp report path: used
- `gpu-fit-validate` real CUDA execution: avoided
- `gpu-fit-validate` contract: mocked handler only
- Handlers moved: no
- Command inventory changed: no, still 51
- Runtime behavior changed: no
- Cloud used: no
- Training started: no
- Phase B started: no
- Model quality claim: none
- SFT-positive expected: 0

## 2. Commands Covered

- `hardware-validate`
- `gpu-fit-validate`

## 3. `hardware-validate` Contract Status

The tests execute `hardware-validate` through subprocess with a temporary `--report-path`.

Locked behavior:

- JSON schema: `hardware_readiness_v1`
- stable report keys are present
- `quality_claim: none`
- `training_readiness_claim: none`
- `hardware_ready_for_training: false`
- text marker: `HARDWARE VALIDATION`
- text markers include `quality_claim`/`report_path` evidence
- report is written to the temporary report path
- default report path is not touched by the targeted tests

The command may return `0` or `1` depending on actual hardware evidence. The tests require the exit code to match the report's `ok` value.

## 4. `gpu-fit-validate` Safe Contract Status

The tests do not run real `gpu-fit-validate` through subprocess.

Instead, they call the existing handler with a fake `train.run_gpu_constrained_fit_validation` module-level dependency. This locks JSON/text/report-path/no-overclaim behavior without allocating CUDA memory, running model/optimizer fit attempts, writing checkpoint artifacts, or running short validation.

Locked behavior:

- JSON schema: `gpu_constrained_fit_validation_v1`
- text marker: `GPU CONSTRAINED-FIT VALIDATION`
- `quality_claim: none`
- `pretraining_readiness_changed: no`
- temporary report path is honored
- direct subprocess execution remains disallowed by the test fixture

## 5. Execution Summary

- `hardware-validate` executed: yes, bounded subprocess only with temporary report path
- `gpu-fit-validate` executed as real CUDA fit: no
- `gpu-fit-validate` execution mode: mocked handler only

## 6. No-Overclaim Checks

Tests reject output claiming:

- Phase B started
- training started
- model quality proven
- training ready
- `ready_for_training: true`

## 7. What Remains Before Extraction

`hardware-validate` is now ready for a narrow wrapper extraction package.

`gpu-fit-validate` should remain in `run.py` until a future package adds a safer extraction plan and import-safety tests around a `cli.commands.hardware` module. It remains high risk because real execution can allocate CUDA memory and write checkpoint evidence.

## 8. Next Recommended Package

`extract hardware-validate wrapper only`

Do not extract `gpu-fit-validate` in that same package.
