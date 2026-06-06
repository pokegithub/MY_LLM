# Cloud Qwen2.5-Coder-7B Evidence Report v1

## 1. Executive Verdict

- 7B cloud readiness: passed.
- 7B hidden eval: 15/17 attempted items passed.
- Model quality proven: no.
- Training ready: no.
- SFT-positive: 0.
- Phase B: not started.
- Quality claim: none.

This report records evidence synced from the stopped Lightning AI RTX 6000 cloud session. It is evidence for backend readiness and one hidden-eval seed run only. It is not a full bakeoff, not training evidence, and not proof of general model quality.

## 2. Environment Summary

- Cloud environment: Lightning AI Studio.
- GPU: RTX PRO 6000 Blackwell.
- Observed VRAM: about 95 GB.
- Model path used in cloud: `run_artifacts/local_models/candidates/open-dense-7b`.
- Model files are generated/downloaded artifacts and are not committed to the repository.

## 3. 7B Provisioning Summary

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Observed download size: about 14.2 GB.
- Cloud-local model path: `run_artifacts/local_models/candidates/open-dense-7b`.
- Backend config path: `configs/backend_cloud_qwen2_5_coder_7b.json`.
- Local repo status: the config is tracked source; the model payload remains outside committed source.

## 4. Evaluation Summary

### Candidate Readiness Smoke

- model_load: passed.
- generation: passed.
- parse_status: passed.
- schema_status: passed.
- candidate_valid: true.
- verifier_status: passed.
- quality_claim: none.

### Agent Backend Smoke

- backend_kind: `local_transformers_in_process`.
- configured_model: `./run_artifacts/local_models/candidates/open-dense-7b`.
- tokenizer_load: passed.
- model_load: passed.
- generation_status: passed.
- parse_status: passed.
- schema_status: passed.
- candidate_valid: true.
- quality_claim: none.

### Hidden Eval Seed Run

- Command used in cloud: `python run.py hidden-eval-seed-run --hidden-eval-backend-config configs/backend_cloud_qwen2_5_coder_7b.json`.
- seed_total: 19.
- attempted: 17.
- passed: 15.
- failed: 2.
- blocked: 0.
- backend_routed: 2.
- retrieval_routed: 12.
- verifier_reached: 14.
- verifier_passed: 12.
- model_id_or_path: `./run_artifacts/local_models/candidates/open-dense-7b`.
- private_leakage: false.
- quality_claim: none.

## 5. Remaining Failures

### hidden_code_repair_001

- Category: `coding_repair_loop_success`.
- Route: `coding_agent_backend`.
- Status: `verification_failed`.
- Evidence: the 7B model chose the wrong slug repair behavior and the verifier correctly rejected it.
- Interpretation: this is a real backend-routed coding repair failure. It must not be counted as model quality proof or SFT-positive evidence.

### hidden_retrieval_002

- Category: `retrieval_grounded_qa`.
- Route: `local_lexical_retrieval`.
- Observed issue: `answer_support_quality: contradicted` and `contradiction_detected: true`, while the surface status remained `answered_with_citations`.
- Required correction: contradicted retrieval answers must not be classified as normal supported answers.
- Local package action: this maintenance package downgrades contradicted cited answers to `abstained_due_to_contradiction` and keeps verifier validity false.

## 6. Training Decision

- No training now.
- No SFT now.
- No DPO, RLVR, distillation, or model-weight improvement now.
- SFT-positive remains 0.
- Hidden eval, fixture, smoke, exact-tool, and verifier-failure traces must not become SFT-positive training data.
- Legal and governance blockers still prevent serious training readiness claims.

## 7. Recommended Next Actions

1. Keep the contradiction status fix in place and rerun hidden retrieval locally.
2. Keep `repo-assist-eval` robust when the local retrieval index is missing.
3. Inspect the repair-loop prompt and critique path later for `hidden_code_repair_001`.
4. Consider 14B only after cheap local retrieval/backend fixes and explicit budget review.
5. Do not start Phase B or training from this evidence alone.
