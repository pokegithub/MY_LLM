# Professional Repository Audit v1

## 1. Executive Verdict

Overall maturity score: 72/100.

The repository is stronger than a typical experimental research repo because it has explicit fail-closed behavior, truthfulness gates, source-policy documents, hidden-eval privacy boundaries, strict trajectory-quality filtering, and a broad test suite. It is not yet company-grade handoff clean because the command surface is concentrated in one very large `run.py`, several subsystems mix report generation with core logic, root-level training/eval modules remain large and old-style, generated artifact roots are huge, and recent report/checkpoint files are still untracked.

Current direction: safe to pause cloud and do professionalization cleanup locally. Do not train. Do not start Phase B.

Top 10 strengths:

1. Strong fail-closed stance around backend, verifier, trajectory-quality, retrieval, and cloud safety.
2. 103 tracked Python files compile under `compile-source`.
3. Full test suite passes: 237 passed, 1 skipped.
4. Hidden/private eval assets are explicitly separated from public reports.
5. `trajectory-quality-audit` remains strict: `sft_positive: 0`.
6. `repo-assist-eval` passes 23/23 with `uses_web: false` and `uses_model_backend: false`.
7. `.gitignore` excludes large generated roots such as `run_artifacts/`, `data_cache/`, checkpoint roots, and quantized outputs.
8. Backend/candidate/retrieval/source-policy contracts are documented with markdown/json companions.
9. Cloud path is guarded and currently deferred.
10. Recent repair-loop and contradiction-handling reports are explicit about not proving model quality.

Top 10 concerns:

1. `run.py` is 2704 lines and mixes CLI parsing, output formatting, command dispatch, report writing, and orchestration glue.
2. Several large modules are handoff risks: `train.py` 2364 lines, `download_data.py` 1749, `agent/backend.py` 1697, `eval_harness/hidden_seed_runner.py` 1160, `retrieval/local_repo.py` 1105.
3. Broad `except Exception` appears in many source files; many are acceptable optional/runtime boundaries, but some data/download/eval paths should be tightened later.
4. Generated/large ignored roots are massive: `data_cache` about 13.5GB, `distill_checkpoints` about 11.9GB, `.venv` about 7.8GB, `run_artifacts` about 5.75GB.
5. Recent accepted reports are untracked and should be intentionally committed or archived before cloud work.
6. Root-level modules mix old pipeline stages with newer Phase A agent/retrieval governance work.
7. `reports/` is becoming a flat chronological pile; useful now, but it will become hard to navigate.
8. Config examples and active smoke configs are close in name; this is safe but easy to misuse.
9. Tests are strong but some classes are very large, especially `tests/test_transformers_backend_mvp.py` and `tests/test_cli_truthfulness.py`.
10. `SYSTEM_MAP.md` is authoritative but extremely large; future sync drift is likely unless supported by smaller docs.

## 2. Current Repo Maturity Rating

| Area | Score | Rationale |
| --- | ---: | --- |
| Truthfulness/safety | 86 | Fail-closed design, explicit `quality_claim`, hidden privacy, trajectory filtering, and citation verification are mature. |
| Test coverage/verification | 82 | Full suite passes and covers many safety seams. Some tests are broad/large but useful. |
| Architecture boundaries | 63 | Subsystems exist, but command/report glue is centralized and large. |
| Handoff cleanliness | 60 | Dirty/untracked reports, large generated roots, and root-level legacy modules reduce handoff polish. |
| Configuration hygiene | 68 | Configs are explicit, but smoke/example/active naming can confuse future operators. |
| Artifact hygiene | 58 | Ignoring is mostly correct, but artifact volume is high and flat reports need lifecycle rules. |
| Overall | 72 | Good Phase A safety repo; not yet professional product/research handoff clean. |

## 3. Top-Level Structure Inventory

Tracked inventory from `git ls-files`:

- 182 tracked files.
- 103 tracked Python files.
- 37 tracked JSON files.
- 35 tracked Markdown files.
- 1 tracked JSONL file: `evals/hidden/hidden_eval_seed_set_v1.jsonl`.

Largest top-level generated/ignored roots by current filesystem size:

| Path | Approx Size | Audit Note |
| --- | ---: | --- |
| `data_cache/` | 13502.4 MB | Generated token/data cache; ignored; do not commit. |
| `distill_checkpoints/` | 11861.94 MB | Generated checkpoint root; ignored by `*_checkpoints/`. |
| `.venv/` | 7784.53 MB | Local environment; ignored. |
| `run_artifacts/` | 5754.82 MB | Generated run/model/eval reports; ignored. |
| `quantized/` | 679.21 MB | Generated quantized outputs; ignored. |
| `eval_results/` | 5.15 MB | Generated eval results; ignored. |

Top tracked directory counts:

| Directory | Tracked Count | Notes |
| --- | ---: | --- |
| `tests/` | 30 | Strong safety/test surface. |
| root | 20 | Legacy/training/eval entry modules live at root. |
| `agent/` | 16 | Agent subsystem is reasonably separated. |
| `evals/` | 15 | Hidden/repo-assist eval assets. |
| `model_selection/` | 12 | Candidate and bakeoff planning. |
| `backend_integration/` | 12 | Backend contracts and policies. |
| `core/` | 11 | Core tensor/config/runtime helpers. |
| `retrieval_design/` | 10 | Design and policy docs. |
| `eval_harness/` | 8 | Hidden eval and harness logic. |
| `configs/` | 8 | Smoke/cloud/candidate configs. |

## 4. File/Folder Classification

| Folder/File | Category | Purpose | Placement Quality | Recommendation |
| --- | --- | --- | --- | --- |
| `agent/` | agent subsystem | Verified coding agent, backend, trajectories, verifier wrapper | Good | Keep; later split `backend.py` only after tests. |
| `retrieval/` | retrieval subsystem | Local lexical repo-doc retrieval, citations, answer verifier | Good but dense | Keep; later split index/search/answer/verifier if needed. |
| `eval_harness/` | hidden eval subsystem | Hidden assets, seed runner, comparisons | Good but dense | Keep; later isolate report writing from runner. |
| `model_selection/` | model selection subsystem | Shortlists, readiness, bakeoff plans | Good | Keep. |
| `data_governance/` | governance subsystem | License/source/holdout policy | Good | Keep; do not casually simplify. |
| `backend_integration/` | backend policy/docs | Backend contracts and failure policy | Good | Keep. |
| `cloud/` | cloud automation | Lightning dry-run guardrails | Good | Keep cloud-only logic here. |
| `configs/` | configs | Backend/cloud examples and smoke configs | Acceptable | Later label active vs example configs more clearly. |
| `tests/` | tests | Safety, backend, retrieval, hidden eval, training checks | Strong but some large files | Keep; later split only if behavior-preserving. |
| `reports/` | reports/docs | Phase evidence and checkpoint docs | Useful but flat | Add report lifecycle/archives later. |
| `run.py` | CLI/orchestration | All command routing/output | Works but too large | High-priority professionalization target. |
| root training files | core runtime/training | `train.py`, `data.py`, `model.py`, SFT/DPO/distill/eval | Legacy acceptable | Later move to package only with broad tests. |
| `run_artifacts/` | generated artifacts | reports, trajectories, local models | Ignored | Do not track; prune intentionally later. |
| `data_cache/` | generated artifacts | token/data caches | Ignored | Do not track. |
| `tokenizer_data/` | tracked runtime artifacts | tokenizer required by status/tests | Mixed | Important but artifact-like; do not move without migration. |

## 5. Slop / Code-Quality Findings

| ID | Evidence | Severity | Confidence | Why It Matters | Recommended Action |
| --- | --- | --- | --- | --- | --- |
| P1-001 | `run.py` is 2704 lines; `main()` is 326 lines; many `run_*` functions contain print/report formatting. | High | High | CLI command growth is now the main maintainability bottleneck. | Later split command handlers into `cli/` or `commands/`, preserving exact command behavior. |
| P1-002 | Large source files: `train.py` 2364 lines, `download_data.py` 1749, `agent/backend.py` 1697, `hidden_seed_runner.py` 1160, `retrieval/local_repo.py` 1105. | Medium | High | Harder review, harder ownership, higher regression risk. | Split only after stabilization; start with `run.py` and report formatting. |
| P1-003 | Broad `except Exception` in `download_data.py`, `eval_suite.py`, `data.py`, `agent/backend.py`, `train_tokenizer.py`, `core/checkpoint_io.py`. | Medium | High | Some are intentional fail-closed adapters; some may hide data/eval issues. | Classify each exception block before changing; do not bulk rewrite. |
| P1-004 | `agent/backend.py:621` uses `except (OSError, ValueError): pass` in prompt path rendering. | Low | High | Probably acceptable best-effort prompt rendering, but silent by design. | Leave unless adding debug diagnostics; path validation remains elsewhere. |
| P1-005 | `download_data.py:get_text/load_ds` prints warnings and falls back/continues after source extraction/load errors. | Medium | High | Data pipeline can produce partial corpora; acceptable if intentional, but handoff needs clearer policy. | Add source failure report summary before any cleanup. |
| P1-006 | `core/ops.py` catches all exceptions when importing `flash_attn`. | Low | High | Optional dependency probe; broad catch is acceptable but should be documented as optional acceleration only. | Keep; not slop. |
| P1-007 | `eval_harness/hidden_seed_runner.py` mixes hidden eval execution, retrieval item handling, private target use, report writing, and targeted rerun delta logic. | Medium | High | Sensitive code in one large module makes privacy review harder. | Later split into assets, coding, retrieval, reporting. |
| P1-008 | `retrieval/local_repo.py` mixes indexing, searching, citation validation, answer assembly, claim extraction, and answer verification. | Medium | High | It works, but professional boundaries are blurred. | Later split into `index.py`, `search.py`, `citations.py`, `answers.py`, `claim_support.py`. |
| P1-009 | Config names: `backend_candidate_smoke_7b_example.json`, `backend_cloud_qwen2_5_coder_7b.json`, `backend_smoke_small_candidate.json` are close enough to misuse. | Low | Medium | Cloud/operator mistakes are possible. | Add config README or naming convention before next cloud session. |
| P1-010 | `reports/` has flat phase/checkpoint files; recent reports are untracked. | Medium | High | Handoff readers may not know what is canonical/current. | Add reports index and archive policy; commit/ignore intentionally. |

## 6. Boundary / Hierarchy Findings

| Boundary | Finding | Risk | Recommended Follow-Up |
| --- | --- | --- | --- |
| CLI vs core | `run.py` owns command parsing, output, JSON report writing, and many command implementations. | High maintainability risk | Extract command modules in a later behavior-preserving pass. |
| Retrieval | Retrieval design docs are separate, but implementation is consolidated into one dense module. | Medium | Split after claim-support tests are locked. |
| Hidden eval | Privacy-sensitive evaluation and public report formatting are in one module. | Medium | Split reporting from private target execution later. |
| Cloud | Cloud logic is mostly isolated under `cloud/`; good boundary. | Low | Keep cloud-only code out of local runtime. |
| Training/governance | Training commands exist but preflight/governance blockers are explicit. | Low to medium | Do not delete training files; professionalize docs and blockers. |
| Reports/docs | Reports are useful evidence but lack index/lifecycle. | Medium | Add `reports/README.md` or archive structure in a docs cleanup pass. |

## 7. Config / Report / Test Findings

Config:

- `configs/` is small and mostly understandable.
- 7B cloud config exists: `configs/backend_cloud_qwen2_5_coder_7b.json`.
- 7B/14B candidate example configs are tracked examples, not proof of local availability.
- Recommendation: add a short config inventory doc before cloud.

Reports:

- Tracked reports include cloud 7B evidence, Phase A checkpoint, targeted rerun workflow, repair diagnostics, repair critique.
- Untracked accepted reports currently visible before this audit: `post_repair_targeted_rerun_checkpoint_v1.*`, `pre_cloud_batch_plan_v1.*`.
- Recommendation: commit accepted reports or create an explicit archive/index.

Tests:

- 237 passed, 1 skipped.
- Large test classes are acceptable for now, but `tests/test_transformers_backend_mvp.py` and `tests/test_cli_truthfulness.py` should eventually be split by behavior area.
- No evidence from this audit that tests require cloud/internet unexpectedly.

## 8. Dead / Unused / Generated File Candidates

No source file should be deleted from this audit alone.

| Candidate | Classification | Evidence | Recommendation |
| --- | --- | --- | --- |
| `run_artifacts/` | definitely generated/cache | ignored by `.gitignore`; 5754.82 MB | Do not track; prune only by explicit artifact cleanup pass. |
| `data_cache/` | definitely generated/cache | ignored; 13502.4 MB | Do not track. |
| `distill_checkpoints/` | definitely generated/cache | ignored by `*_checkpoints/`; 11861.94 MB | Do not track; optional disk cleanup later. |
| `quantized/` | definitely generated/cache | ignored; 679.21 MB | Do not track. |
| `.venv/` | local environment | ignored; 7784.53 MB | Do not track. |
| `loss_history.json` | possibly generated but tracked | root file; `train.py` writes loss history | Needs review before deciding whether tracked artifact belongs in repo. |
| `tokenizer_data/encoder.json`, `merges.json` | artifact-like but important tracked runtime assets | status/tests depend on tokenizer presence | Do not touch without migration plan. |
| `reports/post_repair_targeted_rerun_checkpoint_v1.*` | accepted report artifacts, untracked | `git status --short` | Commit or intentionally archive. |
| `reports/pre_cloud_batch_plan_v1.*` | accepted report artifacts, untracked | `git status --short` | Commit or intentionally archive. |
| `infinite_improver.py`, `alignment.py`, `sft_trainer.py`, `distillation_trainer.py` | not dead | referenced by `run.py` and docs | Do not delete; training remains blocked by readiness/governance, not by absence of code. |

## 9. Professional Target Layout Proposal

Do not move files yet. Proposed eventual layout:

```text
agent/
core/
retrieval/
eval_harness/
model_selection/
data_governance/
configs/
tests/
reports/
docs/
cloud/
scripts/
training/
cli/
run.py
```

Standards to adopt:

- Keep `run.py` as a thin dispatcher only.
- Move command implementations to `cli/commands/`.
- Put cloud-only scripts under `cloud/` only.
- Put generated artifacts only under `run_artifacts/`, `data_cache/`, checkpoint roots, or explicitly ignored output roots.
- Use `reports/<topic>/...` or `reports/archive/...` once reports exceed the current flat set.
- Keep JSON companions only when machine-readable status is useful.
- Use consistent schemas ending in `_vN`.
- Keep test helpers under `tests/helpers/`.
- Keep hidden private targets out of public summaries and training exports.

## 10. Prioritized Cleanup Roadmap

### Pass 2: Zero-Risk Hygiene

Objective: make the working tree and reports intentionally clean.

- Decide whether to commit or archive accepted untracked reports.
- Verify no model files are staged.
- Add a reports index only if useful.
- Confirm `.gitignore` still protects `run_artifacts/`, `data_cache/`, checkpoint roots, quantized outputs, and model files.

Risk: low.  
Tests: `git status --short`, `git diff --check`, `python run.py compile-source`, full pytest.  
Cloud/GPU: no.

### Pass 3: Report / Docs Cleanup

Objective: make evidence easy to navigate.

- Add `reports/README.md` or `reports/index_v1.md`.
- Consolidate which Phase A checkpoint reports are canonical.
- Sync `SYSTEM_MAP.md` after report-index decisions.
- Add config inventory notes for future cloud.

Risk: low.  
Tests: docs-only plus full verification.  
Cloud/GPU: no.

### Pass 4: Code Quality Cleanup

Objective: reduce source maintenance risk without changing behavior.

- Split `run.py` command handlers only after snapshot tests or CLI truthfulness tests cover every moved command.
- Classify broad exceptions into optional dependency, fail-closed adapter, best-effort artifact, and suspicious data/eval cases.
- Extract repeated JSON report helper patterns only if behavior stays identical.
- Keep safety scaffolding intact.

Risk: medium.  
Tests: full suite, targeted CLI tests, hidden eval, trajectory audit.  
Cloud/GPU: no.

### Pass 5: Boundary Cleanup

Objective: clean sensitive module boundaries.

- Split `eval_harness/hidden_seed_runner.py` into private asset execution, retrieval item execution, and public report writing.
- Split `retrieval/local_repo.py` into index/search/citation/answer/claim-support modules.
- Keep hidden/private and retrieval source-policy tests green.

Risk: medium to high.  
Tests: hidden eval assets, retrieval MVP, repo-assist eval, full suite.  
Cloud/GPU: no.

### Pass 6: Final Pre-Cloud Professional Checkpoint

Objective: prepare handoff before future cloud batch.

- Clean git status.
- Full command inventory.
- Full verification set.
- Explicit cloud batch checklist.
- Confirm no model files staged.

Risk: low.  
Tests: full verification.  
Cloud/GPU: no.

## 11. Immediate Next Recommended Package

Recommended next package: `reports index and accepted-artifact staging hygiene`.

Scope:

- Create a concise `reports/index_v1.md`.
- List canonical current reports and which are historical.
- Decide whether the accepted untracked checkpoint reports should be committed.
- Do not refactor source.
- Do not delete reports.
- Run full verification.

## 12. Things Not To Touch Casually

- `agent/trajectory_quality.py`: protects SFT-positive filtering.
- `evals/hidden/private/hidden_eval_private_targets_v1.json`: private eval target data; never expose in summaries/training.
- `eval_harness/hidden_seed_runner.py`: privacy-sensitive; refactor only with tests.
- `retrieval/local_repo.py`: source-policy/citation/claim-support behavior; split only after tests.
- `agent/backend.py`: structured candidate contract, output normalization, safe path behavior, backend fail-closed logic.
- `agent/orchestrator.py`: verifier authority, rollback, repair-loop diagnostics.
- `backend_integration/*`: backend truthfulness contracts.
- `retrieval_design/*`: source policy and citation contracts.
- `data_governance/*`: legal/governance blockers.
- `cloud/lightning_ai/*`: budget and no-paid-overage guardrails.
- `.gitignore`: broad ignore patterns protect large/generated artifacts; change only narrowly.

## 13. Verification Results

Commands run:

- `git status --short`: showed `SYSTEM_MAP.md` modified and accepted untracked report files before audit.
- `git ls-files`: 182 tracked files.
- `git ls-files -o --exclude-standard`: accepted untracked reports only before this audit.
- Static scans for broad exceptions, TODO/stub markers, line counts, function/class sizes, top-level artifact sizes, and reference checks.
- `python run.py status`: passed; no pretrain/SFT/DPO/improved/eval result artifacts found.
- `python run.py deps`: passed.
- `python run.py compile-source`: passed; `scanned: 103`.
- `python run.py repo-assist-eval`: passed; 23/23, supported cited 14, unsupported abstained 9, uses_web false, uses_model_backend false.
- `python run.py hidden-eval-seed-run`: completed; 17 attempted, 13 passed, 0 failed, 4 blocked, private_leakage false, quality_claim none.
- `python run.py trajectory-quality-audit`: completed; scanned 1295, `sft_positive: 0`, rejected_positive 1295.
- `python -m pytest tests -q`: 237 passed, 1 skipped.
- `python -m compileall -q .`: passed.

Not run:

- Cloud commands.
- Model downloads.
- Train/SFT/DPO/RLVR/distillation.
- 14B execution.

Runtime behavior changed: no. This audit only creates report artifacts.

