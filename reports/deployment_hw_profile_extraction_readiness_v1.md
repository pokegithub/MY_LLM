# Deployment / HW Profile Extraction Readiness v1

## 1. Executive Verdict

`deployment-info` is ready for wrapper extraction: yes.

`hw-profile` is ready for wrapper extraction: yes, with one small caution: it is slightly more machine-sensitive because it imports `torch`, inspects CUDA availability, and reads the active hardware profile.

Recommended next package: extract both commands together into `cli/commands/deployment.py`, while preserving parser choices, dispatch keys, command count, JSON contracts, and text markers.

No handlers were moved in this package.

## 2. Current Command Contract Status

| Command | JSON Mode | Text Markers | Contract Coverage | Truthfulness Guard |
| --- | --- | --- | --- | --- |
| `deployment-info` | `deployment_tiers_v1` | `DEPLOYMENT TIERS`, `[2gb_edge]`, `[8gb_laptop]`, `[16gb_workstation]`, `external runtimes   : unverified` | `tests/test_cli_informational_command_contracts.py` | `quality_claim: none`, no training/Phase B/model-quality overclaim |
| `hw-profile` | `hardware_profile_v1` | `STEP 0: Hardware Profile` | `tests/test_cli_informational_command_contracts.py` | `training_started: false`, `phase_b_started: false`, `cloud_used: false`, `model_quality_claim: none` |

The command inventory remains frozen at 51 by `tests/test_cli_command_inventory.py`.

## 3. Implementation Dependency Map

### `run_deployment_info`

- Current handler: `run.py::run_deployment_info`
- Output modes: JSON via `_emit_json`; text via `print_banner` and `print`
- JSON schema: `deployment_tiers_v1`
- Helper dependencies: `_emit_json`, `print_banner`
- External imports inside handler: `config.available_deployment_tiers`, `config.build_deployment_report`
- Global dependencies: `OUTPUT_JSON`
- Side effects: stdout only
- Report writing: none
- Runtime-sensitive dependencies: none beyond config module

### `run_hw_profile`

- Current handler: `run.py::run_hw_profile`
- Output modes: JSON via `_emit_json`; text via `print_banner` and `hardware_profiles.show_profile()`
- JSON schema: `hardware_profile_v1`
- Helper dependencies: `_emit_json`, `print_banner`
- External imports inside handler: `torch`, `hardware_profiles`, `config.hardware_profile_cfg`
- Global dependencies: `OUTPUT_JSON`, `ValidationError` for profile load error handling
- Side effects: stdout only; text mode also uses the existing hardware profile logger through `hardware_profiles.show_profile()`
- Report writing: none
- Runtime-sensitive dependencies: `torch` and machine CUDA state

## 4. Coupling / Boundary Audit

| Area | `deployment-info` Coupling | `hw-profile` Coupling | Notes |
| --- | --- | --- | --- |
| Training logic | none | none | No train/SFT/DPO/RLVR paths are touched. |
| Checkpoint files | none | none | Neither command reads model checkpoints. |
| Hidden eval | none | none | No hidden target access. |
| Retrieval | none | none | No retrieval index or citation path. |
| Backend/parser normalization | none | none | No candidate schema or backend parser dependency. |
| Trajectory quality | none | none | Does not affect SFT-positive eligibility. |
| Cloud scripts | none | none | No cloud commands or downloads. |
| Model downloads | none | none | No provisioning behavior. |
| Runtime config globals | low | low | Both use the process-level `OUTPUT_JSON`; `hw-profile` reads active hardware profile config. |
| Hardware profile config | none | medium-low | `hw-profile` loads a local profile and inspects torch/CUDA state. |

## 5. Proposed Target Module Layout

Recommended module:

`cli/commands/deployment.py`

Recommended functions:

- `build_deployment_info_report() -> dict`
- `run_deployment_info(output_json: bool, emit_json) -> None`
- `build_hw_profile_report() -> dict`
- `run_hw_profile(output_json: bool, emit_json) -> None`

Why one module: both commands are informational environment/deployment-readiness surfaces, and one cohesive module avoids two tiny files while keeping them away from training/cloud/backend/retrieval code.

Import rule for extraction: keep `torch` and `hardware_profiles` imports lazy inside `build_hw_profile_report()` or `run_hw_profile()` so importing the module does not inspect hardware or require command execution.

## 6. Risk Rating

| Command | Risk | Reason |
| --- | --- | --- |
| `deployment-info` | low | Self-contained, config-only, no machine-specific values, JSON/text contracts exist. |
| `hw-profile` | low-medium | JSON/text contracts exist, but output depends on torch/CUDA/profile load. Extract safely by preserving lazy imports and avoiding machine-specific assertions. |

## 7. Required Safety Tests

Already protecting extraction:

- Command inventory remains 51.
- Parser choices and dispatch keys match.
- `deployment-info` JSON contract.
- `deployment-info` text markers.
- `hw-profile` JSON contract.
- `hw-profile` text marker.
- No truthfulness overclaims in informational command output.
- Low-risk `status`, `deps`, and `compile-source` contracts remain green.

Recommended in the extraction package:

- Import-safety check for `cli.commands.deployment` with no command execution during import.
- Continue asserting no exact GPU name, VRAM, CUDA version, file path, or timestamp values.
- Continue asserting `hw-profile --json` is valid JSON on CPU/no-CUDA machines.

## 8. Extraction Recommendation

Option A is recommended: extract `deployment-info` and `hw-profile` together next into `cli/commands/deployment.py`.

Reason: both are informational deployment/hardware profile commands, both now have JSON/text contracts, neither touches training/cloud/hidden eval/retrieval/backend logic, and the shared module boundary is clear.

Stop condition for the extraction package: if import safety or contract tests fail, do not continue into broader command movement.

## 9. Artifact Safety

Read-only artifact checks before creating this report showed:

- `git diff --name-only`: empty
- `git diff --cached --name-only`: empty
- `git ls-files run_artifacts data_cache checkpoints sft_checkpoints dpo_checkpoints distill_checkpoints improved_checkpoints quantized eval_results`: empty
- No `run_artifacts/` files staged.
- No model files staged.
- No cache/checkpoint roots staged.
- No generated roots staged.

## 10. Verification Results

Verification was run after this report was created; see final package response for exact command results.

Expected invariant for this package:

- Handlers moved: no
- Runtime behavior changed: no
- Cloud used: no
- Training started: no
- Phase B started: no
- Model quality claim: none
- SFT-positive: 0

## 11. Next Recommended Package

`extract deployment-info/hw-profile wrappers`

Scope for that package: create `cli/commands/deployment.py`, move only the two informational handlers and pure report-building helpers, update `run.py` dispatch to call wrappers, keep command inventory at 51, and preserve all JSON/text contracts.
