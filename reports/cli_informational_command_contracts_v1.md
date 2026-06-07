# CLI Informational Command Contracts v1

## Summary

- Commands covered: `deployment-info`, `hw-profile`
- JSON contracts locked: yes
- Text markers locked: yes
- Handlers moved: no
- Parser/dispatch changed: no
- Command inventory changed: no, still 51
- Runtime behavior changed: yes, limited to `hw-profile --json`
- Cloud used: no
- Training started: no
- Phase B started: no
- Model quality claim: none

## Contract Notes

`deployment-info` now has a test contract for its structured JSON output and stable text markers. The tests check the schema, quality claim, deployment tier reports, lower-bound estimate scope, unverified external runtimes, and absence of broad model-quality/training/Phase B claims.

`hw-profile` now emits structured `hardware_profile_v1` JSON when invoked with `--json`. Normal text mode remains marker-compatible and still prints `STEP 0: Hardware Profile`.

## Not Tested

- No handler extraction.
- No cloud command execution.
- No model download or model loading.
- No training, SFT, DPO, RLVR, distillation, or Phase B transition.
- No machine-specific hardware values such as GPU name, CUDA version, VRAM, paths, or timestamps.
