# Local Small-Model Stop/Go Memo v1

Date: 2026-05-25

## Decision

Stop local small-model escalation for now on the RTX 2050 path.

## Evidence

- Qwen2.5-Coder-0.5B loaded and generated through `local_transformers_in_process`.
- Qwen2.5-Coder-0.5B passed structured-output smoke after strict markdown-fence normalization, but passed `0` backend-routed hidden coding verifier items.
- Qwen2.5-Coder-1.5B was provisioned locally and loaded/generated, but backend smoke failed structured candidate parsing and passed `0` backend-routed hidden coding verifier items.
- Backend-routed hidden coding verifier pass count across these small-model smokes: `0`.
- Trajectory-quality audit still reports `sft_positive: 0`.
- Qwen2.5-Coder-3B was not downloaded because RTX 2050 4 GB remains constrained and 3B requires a separate guarded opt-in.
- Full 7B/14B bakeoff is not local RTX 2050 work.

## Next Direction

Use RTX 2050/local CPU for a retrieval/citation MVP over approved local repo-authored docs and manifests. This improves source-grounded truthfulness without depending on bigger local coding models.

## Non-Claims

- No small model is declared useful for verified coding trajectory generation.
- No bakeoff winner is selected.
- No model-quality claim is made.
