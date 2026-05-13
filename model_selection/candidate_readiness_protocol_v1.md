# Candidate Readiness Protocol v1

Purpose: decide whether a candidate model is ready to enter the real hidden-eval bakeoff. This protocol does not select a winner and does not score model quality.

## Gates

1. Model availability gate
   - Pass: local checkpoint path exists.
   - Blocked: candidate slot has no local model path or the path is missing.

2. License/deployment review gate
   - Pass: repo policy permits bakeoff execution for the intended use.
   - Blocked: license or deployment review is pending.

3. Runtime load gate
   - Pass: tokenizer and model load through `local_transformers_in_process`.
   - Fail: runtime missing, model load failed, or incompatible checkpoint.

4. Tokenizer/model generation gate
   - Pass: bounded generation executes.
   - Fail: generation errors or times out.

5. Structured candidate JSON gate
   - Pass: output validates against `structured_candidate_contract_v1`.
   - Fail: malformed JSON, markdown/prose-only output, partial JSON, or schema mismatch.

6. Safe-path/schema validation gate
   - Pass: all edits use workspace-relative safe paths and non-empty replacement content.
   - Fail: unsafe path, parent traversal, absolute path, empty edit, or missing required fields.

7. Tiny verifier gate
   - Pass: schema-valid candidate can enter a small controlled verifier path.
   - Fail: candidate applies but verifier fails.
   - Blocked: no schema-valid candidate exists.

8. Trajectory-quality non-inflation gate
   - Pass: audit does not turn readiness smoke traces into unsupported SFT-positive data.
   - Fail: fixture, backendless, malformed, or blocked traces become positive training data.

9. Hidden-eval readiness gate
   - Pass: mandatory gates are green and private hidden eval artifacts exist.
   - Blocked: any mandatory evidence is missing.

## Truth Limits

- Model availability is not quality.
- Structured output is not code correctness.
- A toy verifier pass is not bakeoff success.
- The shortlist has no winner.
- Full 7B/14B bakeoff has not been executed.
