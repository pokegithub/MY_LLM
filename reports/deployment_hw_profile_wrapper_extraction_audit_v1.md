# Deployment / HW Profile Wrapper Extraction Audit v1

## 1. Executive Verdict

The `deployment-info` / `hw-profile` extraction is clean.

- Stale duplicate implementation in `run.py`: no
- Import boundary safe: yes
- Lazy torch import preserved: yes
- Hardware inspected on import: no
- Command inventory: still 51
- JSON schemas changed: no
- Text markers changed: no
- Runtime behavior changed: no
- Cloud used: no
- Training started: no
- Phase B started: no
- Model quality claim: none
- SFT-positive: 0

## 2. Stale Duplicate Check

`run.py` no longer contains:

- `def run_deployment_info`
- `def run_hw_profile`
- `def build_deployment_info_report`
- `def build_hw_profile_report`
- the old direct `DEPLOYMENT TIERS` implementation block
- the old direct `hardware_profile_v1` JSON-building block
- eager `torch` or `hardware_profiles` imports for the hardware profile command

Expected remaining references in `run.py`:

- `from cli.commands.deployment import run_deployment_info as cli_run_deployment_info`
- `from cli.commands.deployment import run_hw_profile as cli_run_hw_profile`
- `"deployment-info": lambda: cli_run_deployment_info(OUTPUT_JSON, _emit_json)`
- `"hw-profile": lambda: cli_run_hw_profile(OUTPUT_JSON, _emit_json)`

## 3. Import-Boundary Check

`cli/commands/deployment.py` does not import `run.py`.

`run.py` imports `cli.commands.deployment` one-way.

No circular import risk is visible.

The module exposes:

- `build_deployment_info_report`
- `run_deployment_info`
- `build_hw_profile_report`
- `run_hw_profile`

## 4. Lazy Hardware Import Verification

Import probe result:

- stdout during import: empty
- stderr during import: empty
- `torch` imported during module import: false
- `hardware_profiles` imported during module import: false
- hardware inspected during module import: false

`torch` and `hardware_profiles` remain lazy inside `build_hw_profile_report()` / text-mode `run_hw_profile()`.

## 5. Contract Preservation Summary

Preserved command contracts:

- `deployment-info --json`: `deployment_tiers_v1`
- `deployment-info` text marker: `DEPLOYMENT TIERS`
- `hw-profile --json`: `hardware_profile_v1`
- `hw-profile` text marker: `STEP 0: Hardware Profile`

Truthfulness fields remain preserved:

- `quality_claim: none`
- `training_started: false`
- `phase_b_started: false`
- `cloud_used: false`
- `model_quality_claim: none`

## 6. `run.py` Line Count / Responsibility Summary

Line counts:

- `run.py` current: 2578
- previous known count after low-risk extraction audit: 2626
- reduction from previous known count: 48
- `cli/commands/deployment.py`: 137
- `cli/commands/status.py`: 224
- `cli/commands/compile_source.py`: 90

Remaining `run.py` responsibilities:

- parser construction
- runtime config application
- dependency gating
- dispatch table
- remaining command wrappers
- high-risk training/eval/data command wrappers
- retrieval/repo-assist command wrappers
- hidden-eval command wrappers
- backend/candidate command wrappers
- trajectory-quality command wrappers
- governance/status command wiring

## 7. Artifact Safety Result

Artifact safety commands showed:

- `git diff --cached --name-only`: empty
- `git ls-files run_artifacts data_cache checkpoints sft_checkpoints dpo_checkpoints distill_checkpoints improved_checkpoints quantized eval_results`: empty
- no `run_artifacts/` files staged
- no model files staged
- no cache/checkpoint roots staged
- no protected generated roots tracked unexpectedly

## 8. Remaining Risks

No behavior risk was found in this extraction audit.

The only ordinary caveat is that `hw-profile` output is machine-sensitive when executed, but module import remains machine-neutral and contract tests avoid exact GPU/VRAM/CUDA assertions.

## 9. Next Recommended Package

`next low-risk informational command readiness audit`

Good candidate group: `hardware-validate` / `gpu-fit-validate`, but only as a readiness audit first because those commands have stronger hardware and report-writing coupling than `deployment-info` / `hw-profile`.

## 10. Verification Results

Verification was run after this report was created; see final package response for exact command results.
