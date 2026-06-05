# Phase A Final Checkpoint v1

Date: 2026-06-05

Purpose: summarize the current Phase A repository state without changing runtime behavior. This report is a documentation/evidence checkpoint only. It does not start Phase B, prove model quality, approve training, grant legal clearance, or create SFT-positive data.

## 1. Executive Verdict

| Item | Verdict | Evidence / Limit |
|---|---|---|
| Phase A status | stabilized | Source compile, repo-assist-eval, hidden eval, trajectory audit, and tests have passing evidence in the current package verification flow. |
| Phase B started | no | `SYSTEM_MAP.md` states Phase B has not started; `repo-assist --query "Has Phase B started?"` returns `No. Phase B has not started.` with citations. |
| Model quality proven | no | Small-model smoke and verifier reachability are integration evidence only, not model-quality proof. |
| SFT-positive data exists | no | `run.py trajectory-quality-audit --json` reported `sft_positive_eligible: 0` at checkpoint creation. |
| Training/SFT/DPO/RLVR ready | no | No SFT-positive dataset, no legally cleared broad training corpus, no model-quality checkpoint, and no Phase B start. |
| 7B/14B bakeoff run | no | Candidate/bakeoff docs exist, but no full 7B/14B bakeoff has executed. |
| RTX 2050 local model escalation | stop/pause | RTX 2050 4GB remains useful for smoke, integration, retrieval, and repo-assist, not serious 7B/14B local bakeoff. |
| Strongest current local capability | retrieval/citation/repo-assist truthfulness | Local lexical retrieval, citations, claim support, abstention, and direct repo-status answers are the most useful stabilized path. |

Executive summary: Phase A is stable enough to stop escalating small local coding models on RTX 2050. The repo should either continue local retrieval/repo-assist usability work or move to a reviewed RTX 6000/cloud 7B/14B candidate-readiness/bakeoff path if stronger hardware is available.

## 2. Verified Capabilities

| Capability | Status | Evidence command / artifact | Truth limit |
|---|---|---|---|
| Source-scoped compile | verified | `run.py compile-source` | Compiles tracked Python sources only; not a replacement for full `compileall`. |
| Dependency check | verified | `run.py deps` | Checks declared runtime deps; not training readiness. |
| Hidden eval seed runner | verified | `run.py hidden-eval-seed-run` | Narrow seed execution only; not full bakeoff or training data. |
| Retrieval index build | verified | `run.py retrieval-index-build` | Local lexical repo-doc index only; no web/vector/memory. |
| Retrieval answer | verified | `run.py retrieval-answer --query "What is the backend failure contract?"` | Cited extractive answer; lexical support is not full semantic truth. |
| Repo-assist | verified | `run.py repo-assist --query "Has Phase B started?"` | User-facing local repo QA; no model/backend generation. |
| Repo-assist eval | verified | `run.py repo-assist-eval` | Manual query pack: 23/23 passed at checkpoint creation. |
| Citation validation | verified | `retrieval/local_repo.py`, `retrieval_design/citation_contract_v1.*` | Validates source/locator/hash; does not prove broad truth. |
| Claim-level lexical support | verified | `retrieval/local_repo.py`, hidden retrieval reports | Lexical/direct support only; no semantic entailment engine. |
| Trajectory-quality audit | verified | `run.py trajectory-quality-audit` | Final verification run: 947 scanned, SFT-positive 0. |
| Backend smoke path | verified as integration surface | `agent-backend-smoke`, backend integration docs | Smoke/load/parse evidence only; no model-quality claim. |
| Qwen small-model smoke | verified as limited evidence | `model_selection/local_small_model_stop_go_memo_v1.md` | Qwen 0.5B/1.5B did not pass backend-routed hidden coding verifier items. |
| SYSTEM_MAP sync | present | `SYSTEM_MAP.md` | Manifest, not runtime evidence by itself. |

## 3. Small-Model Evidence Summary

- Qwen2.5-Coder 0.5B loaded and generated through `local_transformers_in_process`.
- Qwen2.5-Coder 0.5B passed structured-output smoke after strict markdown-fenced JSON normalization, but passed 0 backend-routed hidden coding verifier items.
- Qwen2.5-Coder 1.5B was provisioned and could load/generate, but did not improve hidden coding verifier pass evidence; backend-routed hidden verifier pass count remained 0.
- SmolLM2-135M appears as an allowlisted early small-candidate/provisioning artifact, but it is not evidence of a useful verified coding trajectory generator.
- Qwen2.5-Coder 3B was skipped by policy unless separately guarded because RTX 2050 4GB is constrained.
- These results validate backend plumbing and truthfulness handling; they do not prove model quality, coding ability, or training-readiness.
- Decision: pause local small-model escalation on RTX 2050.

## 4. Retrieval / Source-Grounded Layer Summary

Implemented and verified:

- Local lexical retrieval over approved repo-authored docs/manifests.
- Source-policy exclusions for hidden private targets, fixtures, model dirs, caches, corpora, and broad run artifacts.
- Citation objects with source id, document ref, locator type, locator, support kind, and support snippet hash.
- Citation validation for required fields, source policy, locator resolution, and snippet/hash support.
- Cited answer assembly with abstention for missing/out-of-scope evidence.
- Claim-level lexical support reporting, including unsupported, partial, contradicted, missing-citation, and weak-support cases.
- Hidden source/retrieval eval expansion with private leakage controls.
- `repo-assist` for user-facing repo questions.
- `repo-assist-eval` manual query pack.
- Deterministic direct-answer templates for narrow repo-status, command, SFT/training-status, and truth-limit questions.

Hard limits:

- No web retrieval.
- No vector search.
- No embeddings.
- No semantic memory.
- No model/backend answer generation.
- No full semantic entailment.
- Citation and lexical support are evidence checks, not full truth proof.

## 5. Hidden Eval Status

Latest observed hidden-eval run in the preceding verification flow:

| Metric | Value |
|---|---:|
| seed_total | 19 |
| attempted | 17 |
| passed | 14 |
| failed | 2 |
| blocked | 1 |
| exact_tool_routed | 3 |
| backend_routed | 2 |
| retrieval_routed | 12 |
| retrieval_supported | 8 |
| retrieval_abstained | 3 |
| retrieval_partial | 1 |
| verifier_reached | 13 |
| verifier_passed | 11 |
| private_leakage | false |
| quality_claim | none |

Hidden eval proves only narrow evaluator behavior for the current seed set. It does not prove model quality, does not authorize training, and must not be exported as SFT-positive data.

## 6. Trajectory / Training-Use Status

Latest trajectory audit snapshot from the final verification run:

| Metric | Value |
|---|---:|
| scanned_trajectories | 947 |
| sft_positive_eligible | 0 |
| preference_winner_eligible | 0 |
| preference_loser_eligible | 129 |
| retrieval_memory_eligible | 474 |
| exact_tool_curriculum_only | 252 |
| rejected_for_positive_training | 947 |

Training must not start now because:

- SFT-positive count is 0.
- Preference winners are 0.
- Existing traces are mostly exact-tool, fixture, scripted, backendless, or failure-analysis records.
- Hidden-eval/fixture/smoke traces are explicitly excluded from positive training use.
- No legal/data-governance clearance exists for serious training.

Before training-use data is allowed, the repo needs verified non-fixture model-backed coding successes with meaningful verifier evidence, clean provenance, legal/governance clearance, and strict trajectory-quality acceptance.

## 7. Data Governance and Legal Readiness

| Area | Status |
|---|---|
| legal_clearance_claim | none |
| source review | mostly blocked/needs manual review |
| allowed corpus subset | tiny, repo-policy scoped, not broad legal clearance |
| benchmark holdout policy | present |
| serious training readiness | blocked/restricted |

Important caveat: repo-policy allowance is not legal approval. The metadata says it is not legal advice and not legal clearance. Many sources remain hard blockers due to unknown licenses, benchmark exclusion, unresolved provenance, or training-blocker status.

## 8. Hardware Status

| Hardware path | Status |
|---|---|
| RTX 2050 4GB | validation/smoke/retrieval-usefulness hardware |
| GPU fit validation | useful bounded evidence, not pretraining readiness |
| local 7B/14B bakeoff | not appropriate on this hardware |
| serious training | not practical/ready locally |

RTX 2050 remains valuable for:

- CLI verification.
- Local lexical retrieval.
- Repo-assist.
- Small backend smoke checks.
- Tiny/guarded Transformers runtime experiments.

RTX 2050 should not be used as the local path for serious 7B/14B bakeoff or model training.

## 9. Current Blocker Inventory

| Blocker | Severity | Evidence | Unlocks When | Recommended Timing |
|---|---|---|---|---|
| No SFT-positive data | critical | trajectory audit: SFT-positive 0 | Real verified non-fixture model-backed trajectories exist | Before any SFT |
| No model-quality checkpoint | critical | status shows missing pretrain/SFT/DPO/improved checkpoints | Real checkpoint and eval evidence exist | Before Phase B claims |
| No 7B/14B bakeoff | high | model-selection docs only; no run evidence | Stronger hardware/cloud and reviewed checkpoints available | When RTX 6000/cloud path exists |
| No legal clearance | critical | `legal_clearance_claim: none` | Source review/legal approval and allowed corpus policy mature | Before serious training |
| No full semantic retrieval | medium | lexical/locator support only | Explicit semantic/vector/entailment design and tests exist | Later |
| No web/current source support | medium | repo-assist abstains on current-world questions | Approved web/current-source policy and retrieval implementation exist | Later, if needed |
| No production serving backend | high | serving remains fail-closed until real backend configured | Reviewed runtime/backend and deployment evidence exist | After model-quality evidence |
| RTX 2050 constraint | high | 4GB hardware classification | RTX 6000/cloud/stronger hardware available | Before serious bakeoff/training |

## 10. Next-Path Decision Matrix

| Path | Value | Risk | Hardware Need | Readiness | Recommendation |
|---|---|---|---|---|---|
| A. Continue local repo-assist/retrieval usefulness | high for local truthfulness and usability | low | RTX 2050/CPU enough | ready | recommended if staying local |
| B. Prepare RTX 6000/cloud 7B/14B bakeoff | high for model selection evidence | medium/high cost and setup risk | RTX 6000 or cloud GPU | not local-ready, protocol-ready | recommended only with stronger hardware/cloud |
| C. Legal/data governance cleanup | high for future training safety | medium process risk | CPU/local enough | partially ready | recommended before serious training |
| D. Build semantic/vector retrieval later | medium future value | medium truthfulness and complexity risk | CPU/GPU optional depending design | not needed now | later |
| E. Start training now | low/negative | critical truth/legal/model risk | insufficient locally | not ready | not recommended |

## 11. Final Recommendation

Phase A local path is stabilized.

Stop small-model escalation on RTX 2050 for now. Do not start Phase B, SFT, DPO, RLVR, model fine-tuning, production serving, or 7B/14B local bakeoff on this laptop path.

Recommended next action depends on available hardware:

- If staying local on RTX 2050: continue repo-assist/retrieval usability, source-policy hardening, and governance cleanup.
- If using RTX 6000 or cloud GPU: run the candidate readiness and 7B/14B bakeoff path with reviewed checkpoints and no training claims.
- If preparing real training: legal/governance review and eligible trajectory/data pipeline must come first.

Non-claims remain unchanged:

- Phase B has not started.
- Model quality is not proven.
- SFT-positive remains 0.
- Training is not ready.
- Repo-assist uses no web and no model backend generation.
