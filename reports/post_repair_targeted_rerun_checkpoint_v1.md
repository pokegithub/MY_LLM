# Post-Repair Targeted Rerun Checkpoint v1

## 1. Executive Verdict

- Targeted rerun completed: yes.
- Baseline used: `E:\MY_LLM\run_artifacts\hidden_eval_runs\hidden_eval_seed_run_1780723827_fb003cb2\summary.json`.
- New targeted run: `E:\MY_LLM\run_artifacts\hidden_eval_runs\hidden_eval_seed_run_1780724816_2ed25a59\summary.json`.
- Seeds rerun: `hidden_code_repair_001`, `hidden_retrieval_002`.
- Improved: 0.
- Regressed: 0.
- Unchanged: 2.
- Privacy held: yes, `private_leakage: false`.
- SFT-positive changed: no, trajectory audit remains `sft_positive: 0`.
- Training recommended: no.
- Phase B started: no.
- Model quality proven: no.

The local repair-loop prompt/critique work improved diagnostics and future repair feedback, but this targeted local rerun did not convert either known non-passed seed into a pass.

## 2. Targeted Rerun Summary

Command used:

```powershell
python run.py hidden-eval-rerun-failed --previous-summary "E:\MY_LLM\run_artifacts\hidden_eval_runs\hidden_eval_seed_run_1780723827_fb003cb2\summary.json" --hidden-eval-seed-id hidden_code_repair_001 --hidden-eval-seed-id hidden_retrieval_002
```

| Seed | Previous Status | New Status | Delta | Notes |
| --- | --- | --- | --- | --- |
| `hidden_code_repair_001` | `blocked_unverified` | `blocked_unverified` | unchanged | Local Qwen 0.5B path still blocked before verifier due malformed candidate output. |
| `hidden_retrieval_002` | `blocked_unverified` | `blocked_unverified` | unchanged | Contradiction is handled safely as abstention, not a false supported answer. |

Delta report:

- `improved_count: 0`
- `regressed_count: 0`
- `unchanged_count: 2`
- `previous_passed_count: 0`
- `new_passed_count_for_rerun_subset: 0`
- `private_leakage: false`
- `quality_claim: none`
- `training_executed: false`
- `phase_b_started: false`

## 3. Repair-Loop Finding

`hidden_code_repair_001` result:

- Final status: `blocked_unverified`.
- Verifier reached: false.
- Verifier passed: false.
- Failure category: `malformed_candidate_output`.
- Repair diagnostics exist: true.
- Repair prompt/critique fields in this item: false, because the local backend blocked on malformed candidate output before a verifier-backed repair prompt could be used.
- Failed edits rolled back: no applied winning candidate; training use remains disallowed.
- `training_use_allowed: false`.
- `quality_claim: none`.

Interpretation: repair-loop diagnostics and critique feedback are available in the agent, but this local 0.5B targeted seed did not reach a successful repair-verifier cycle. This is not a training-positive trace.

## 4. Retrieval Contradiction Finding

`hidden_retrieval_002` result:

- Final status: `blocked_unverified`.
- Verifier reached: true.
- Verifier passed: false.
- `answer_status: abstained_due_to_contradiction`.
- `answer_support_quality: contradicted`.
- `contradiction_detected: true`.
- `retrieval_supported: false`.
- `retrieval_abstained: true`.
- Citations remain visible: yes, `citation_count: 2`.
- Citations treated as proof: no; `claim_verifier_passed: false`, `claims_contradicted: 1`.
- `training_use_allowed: false`.
- `quality_claim: none`.

Interpretation: the retrieval path fails closed on contradiction. It preserves evidence visibility without turning contradictory evidence into a supported answer.

## 5. Verification Summary

Required verification completed:

- `python run.py status`: passed.
- `python run.py deps`: passed.
- `python run.py compile-source`: passed, `scanned: 103`.
- `python run.py repo-assist-eval`: passed, `23/23`, supported cited `14`, unsupported abstained `9`, `uses_web: false`, `uses_model_backend: false`.
- `python run.py hidden-eval-seed-run`: completed, `17 attempted`, `13 passed`, `0 failed`, `4 blocked`, `private_leakage: false`.
- `python run.py trajectory-quality-audit`: passed, `scanned: 1245`, `sft_positive: 0`.
- `python -m pytest tests -q`: `237 passed, 1 skipped`.
- `python -m compileall -q .`: passed.
- `git diff --check`: passed.
- `git status --short`: clean before this report was written.

## 6. Decision

- Do not train.
- Do not start Phase B.
- Do not claim model quality.
- Do not treat hidden eval traces as SFT-positive.
- Local logic is safer and better instrumented, but the local targeted rerun did not improve these two non-passed seeds.

Next possible paths:

1. Stop here and use the local repo workflow as a truthful retrieval/diagnostic tool.
2. Run one small cloud 7B targeted rerun later, if manually approved, to see whether the better repair feedback helps a stronger model.
3. Consider 14B only after budget and hardware review.
4. Begin legal/eligible-data work before any training discussion.
