# Pre-Cloud Batch Plan v1

## A. Executive Verdict

- Local Phase A is stable enough to pause cloud work.
- Further cloud work should be batched into one future session instead of drip-running individual checks.
- Do not train yet.
- Do not start Phase B.
- Do not claim model quality from existing smoke/eval evidence.
- Current local direction remains: repo-assist, retrieval/citation truthfulness, hidden-eval diagnostics, and safe checkpoint reporting.

## B. Current Verified Local Capabilities

- `retrieval-index-build`: deterministic local repo-doc lexical index build.
- `repo-assist`: local repo-doc assistant with cited answers and abstention for current/external questions.
- `repo-assist-eval`: manual query pack, currently expected to show supported cited answers, unsupported abstentions, `uses_web: false`, and `uses_model_backend: false`.
- Retrieval contradiction handling: contradicted answers downgrade away from normal supported answers, including `abstained_due_to_contradiction`.
- Repair-loop diagnostics: failed repair attempts can expose safe verifier-failure evidence, rollback status, and `training_use_allowed: false`.
- Repair-loop prompt critique: repair attempts receive verifier feedback, failure class, previous-candidate summary, and retry-budget reminders.
- `hidden-eval-rerun-failed`: reruns selected failed/non-passed hidden-eval seeds and writes public-safe delta reports.
- `trajectory-quality-audit`: remains strict; hidden/fixture/smoke traces do not become SFT-positive automatically.
- Cloud 7B evidence report: `reports/cloud_qwen2_5_coder_7b_evidence_v1.md` records synced Qwen2.5-Coder-7B cloud evidence without starting Phase B or training.

## C. Future Cloud Batch Objectives

Run the future Lightning Studio/Linux session in this order:

1. Environment/GPU check.
2. Budget/credit confirmation.
3. Pull latest repo.
4. Verify the 7B backend config.
5. Ensure `Qwen/Qwen2.5-Coder-7B-Instruct` model path exists at `run_artifacts/local_models/candidates/open-dense-7b`, or re-download only after user approval and license/source review.
6. Run 7B `candidate-readiness-smoke`.
7. Run 7B `agent-backend-smoke` with the explicit 7B backend config.
8. Run 7B targeted rerun for:
   - `hidden_code_repair_001`
   - `hidden_retrieval_002`
9. Optionally run full 7B `hidden-eval-seed-run`.
10. Only if budget remains and the user approves, evaluate whether 14B is worth provisioning.
11. Do not train.

## D. Future Cloud Commands

Linux/Lightning Studio command plan only. These commands are for a future manually approved cloud session, not for this local checkpoint.

```bash
cd /path/to/MY_LLM
git status --short
git pull
git status --short
```

```bash
nvidia-smi
python run.py status
python run.py deps
python run.py compile-source
```

Budget guard reminder:

```bash
export MYLLM_CLOUD_MAX_CREDITS=15
export MYLLM_CLOUD_CONFIRM_FREE_CREDIT_ONLY=YES
export MYLLM_CLOUD_ALLOW_PAID_OVERAGE=NO
export MYLLM_CLOUD_CONFIRM_NO_TRAINING=YES
```

Local truthfulness smoke before model work:

```bash
python run.py retrieval-index-build
python run.py repo-assist-eval
python run.py trajectory-quality-audit
```

Verify 7B model/config presence:

```bash
test -f configs/backend_cloud_qwen2_5_coder_7b.json
test -d run_artifacts/local_models/candidates/open-dense-7b
python - <<'PY'
import json
from pathlib import Path
p = Path("configs/backend_cloud_qwen2_5_coder_7b.json")
print(json.dumps(json.loads(p.read_text()), indent=2))
PY
```

7B readiness smoke:

```bash
python run.py candidate-readiness-smoke \
  --candidate-id open_dense_7b_candidate_slot \
  --shortlist-path model_selection/candidate_shortlist_v1.json
```

7B backend smoke:

```bash
python run.py --config configs/backend_cloud_qwen2_5_coder_7b.json --no-env-overrides agent-backend-smoke
```

7B targeted hidden-eval rerun:

```bash
python run.py hidden-eval-rerun-failed \
  --previous-summary run_artifacts/hidden_eval_runs/<latest_full_summary>/summary.json \
  --hidden-eval-backend-config configs/backend_cloud_qwen2_5_coder_7b.json \
  --hidden-eval-seed-id hidden_code_repair_001 \
  --hidden-eval-seed-id hidden_retrieval_002
```

Optional full 7B hidden eval if budget remains:

```bash
python run.py hidden-eval-seed-run \
  --hidden-eval-backend-config configs/backend_cloud_qwen2_5_coder_7b.json
```

Final cloud-session audit:

```bash
python run.py trajectory-quality-audit
python -m pytest tests -q
python -m compileall -q .
git diff --check
git status --short
```

Do not run `train`, `sft`, `dpo`, `distill`, `improve`, RL/RLVR, or any model-weight update command.

## E. Stop Conditions

Stop the cloud session if any of these occur:

- Credits are low, unknown, or paid-overage risk is unclear.
- GPU is not RTX 6000-class or sufficient VRAM for the approved model.
- Repo is dirty unexpectedly after `git pull`.
- 7B model is missing and the user does not approve re-download.
- Any command path attempts or suggests training.
- Candidate readiness fails unexpectedly.
- `agent-backend-smoke` cannot load/generate/validate the configured 7B candidate.
- `private_leakage` is true.
- `SFT-positive` unexpectedly changes without review.
- Hidden private targets appear in any public summary.
- Runtime reports claim model quality, training readiness, or Phase B.

## F. 14B Decision Gate

14B may be considered only if all are true:

- 7B targeted rerun remains insufficient.
- Budget remains after the approved 7B batch.
- The user manually approves 14B evaluation.
- Disk is sufficient.
- Download cost/time is acceptable.
- License/source review is clear enough for local testing.
- No training is started.

14B should not be downloaded or run as a side effect of the 7B batch.

## G. Training Decision

- Training remains blocked.
- SFT-positive is 0.
- Legal/governance is not ready for broad training.
- Hidden eval traces are not training data.
- Failed or blocked hidden traces remain non-training-positive.
- Phase B has not started.

## H. Expected Output To Paste Back

After the future cloud session, paste:

- GPU summary from `nvidia-smi`.
- Credit/budget confirmation and rough credit usage estimate.
- 7B `candidate-readiness-smoke` result.
- 7B `agent-backend-smoke` result.
- 7B targeted rerun delta for `hidden_code_repair_001` and `hidden_retrieval_002`.
- 7B full hidden-eval summary if run.
- `trajectory-quality-audit` summary.
- Any failure report paths.
- Final `git status --short`.

## I. Checkpoint Decision

Cloud is deferred. Local Phase A remains the active phase. The repo is prepared for a future batched cloud session, but this plan does not run cloud, download models, train, start Phase B, or prove model quality.
