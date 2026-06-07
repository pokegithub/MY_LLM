# HW Profile JSON Support v1

## Executive Verdict

`python run.py --json hw-profile` now emits valid machine-readable JSON. The behavior change is intentional and limited to `hw-profile --json`.

## Old Behavior

`hw-profile` accepted the global `--json` flag but emitted the same text banner as normal mode.

## New Behavior

`hw-profile --json` emits a `hardware_profile_v1` payload with hardware/profile status, torch/CUDA availability, device metadata, selected profile data, warnings, and explicit truthfulness fields.

## JSON Schema Keys

- `schema`
- `command`
- `profile_status`
- `torch_available`
- `cuda_available`
- `device_count`
- `devices`
- `selected_profile`
- `active_profile_config`
- `profile`
- `warnings`
- `training_started`
- `phase_b_started`
- `cloud_used`
- `model_quality_claim`
- `quality_claim`

## Text Output Preservation

`python run.py hw-profile` still preserves the stable text marker `STEP 0: Hardware Profile`.

## Truthfulness Limits

- Cloud used: no
- Training started: no
- Phase B started: no
- Model quality claim: none
- SFT-positive expected: 0

This JSON support does not prove model quality, training readiness, deployment readiness, or hardware suitability for larger models.

## Next Recommended Package

Run a small wrapper-extraction readiness audit for `deployment-info` and `hw-profile`, without moving handlers yet.
