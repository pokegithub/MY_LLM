# Deployment / HW Profile Wrapper Extraction v1

## Executive Verdict

`deployment-info` and `hw-profile` wrappers were extracted into `cli/commands/deployment.py`.

The extraction is behavior-preserving: command names, parser choices, dispatch keys, JSON schemas, text markers, and truthfulness fields are unchanged.

## Wrappers Moved

- `run_deployment_info(output_json=False, emit_json=None)`
- `run_hw_profile(output_json=False, emit_json=None)`

## Helpers Moved

- `build_deployment_info_report()`
- `build_hw_profile_report()`

## What Did Not Move

- Parser construction remains in `run.py`.
- Dispatch table remains in `run.py`.
- Training, cloud, hidden-eval, retrieval, backend, and trajectory-quality commands were not moved.
- No command names were changed.

## Lazy Import / Import Safety

`torch` and `hardware_profiles` remain lazy imports inside the hardware profile report/command path. Importing `cli.commands.deployment` does not inspect hardware, does not import `torch`, does not import `hardware_profiles`, and does not print output.

The import-safety test is `tests/test_cli_deployment_command_imports.py`.

## Contract Preservation

Preserved:

- `deployment-info --json`: `deployment_tiers_v1`
- `deployment-info` text marker: `DEPLOYMENT TIERS`
- `hw-profile --json`: `hardware_profile_v1`
- `hw-profile` text marker: `STEP 0: Hardware Profile`
- `quality_claim: none`
- `training_started: false`
- `phase_b_started: false`
- `cloud_used: false`
- `model_quality_claim: none`

## Truthfulness Limits

- Cloud used: no
- Training started: no
- Phase B started: no
- Model quality claim: none
- SFT-positive expected: 0

This extraction does not prove model quality and does not make training ready.

## Next Recommended Package

Run a stale-duplicate extraction audit for `run.py` to confirm no old `deployment-info` or `hw-profile` implementation remains, then consider the next low-risk informational wrapper candidate.
