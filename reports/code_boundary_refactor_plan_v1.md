# Code Boundary Refactor Plan v1

## 1. Executive Verdict

Code-boundary refactor plan created: yes.

Runtime behavior changed: no. This package is planning-only and does not move source files, rename modules, change CLI semantics, alter verifier authority, change retrieval behavior, or start cloud/training work.

`run.py` is the largest immediate handoff risk because it currently owns parser setup, global runtime config application, dependency gating, command dispatch, text/JSON output formatting, report writing, and many command wrappers in one 2704-line file. The safest next move is not extraction yet. The next move should be CLI command inventory/snapshot tests so future extraction can prove command-surface preservation.

Current truth:

- Command surface inventoried: yes.
- Commands inventoried: 51.
- Cloud used: no.
- Training started: no.
- Phase B started: no.
- Model quality claimed: no.
- SFT-positive remains 0.

## 2. Why Refactor Planning Is Needed

The professional repo audit identified strong truthfulness scaffolding but weak handoff boundaries. The issue is not that safety code is verbose; much of it is intentionally explicit. The issue is that several command surfaces and report builders have grown into large mixed-responsibility modules.

Evidence:

- `run.py`: 2704 lines; `main()` is 326 lines; parser choices, arguments, dependency gates, dispatch table, and command wrappers live together.
- `train.py`: 2364 lines; keep last because training behavior is high risk.
- `download_data.py`: 1749 lines; data-source fallback behavior needs careful truthfulness review before movement.
- `agent/backend.py`: 1697 lines; backend schema, output normalization, path safety, provisioning, model load, and smoke reporting are safety-critical.
- `eval_harness/hidden_seed_runner.py`: 1160 lines; hidden/private privacy and public reporting are tightly coupled.
- `retrieval/local_repo.py`: 1105 lines; index/search/citation/answer/claim verification are all together.

The refactor goal is to shrink command-boundary code without weakening fail-closed behavior, hidden-eval privacy, source-policy validation, or trajectory-quality filtering.

## 3. Current `run.py` Command Inventory

Inventory source: `python run.py --help` and the `commands = { ... }` dispatch table in `run.py`.

| Command | Current Handler | Subsystem | Future Target Module | Risk | Safe For First Extraction | Tests Required Before Moving | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `status` | `show_status` | status/deps | `cli/commands/status.py` | Low | Yes, after snapshots | help snapshot, status text/JSON keys | Pure status/reporting boundary. |
| `deps` | `run_deps` | status/deps | `cli/commands/deps.py` | Low | Yes, after snapshots | deps output keys, dependency failure path | Wrapper around dependency checks. |
| `compile-source` | `run_compile_source` | source verification | `cli/commands/compile_source.py` | Low | Yes, after snapshots | tracked-source compile test, bad-source fixture if added | Behavior must keep skipped roots explicit. |
| `audit` | `run_audit` | audit | `cli/commands/audit.py` | Low/Medium | Yes, after snapshots | audit output truthfulness keys | Keep narrow audit semantics. |
| `repo-assist` | `run_repo_assist` | retrieval UX | `cli/commands/repo_assist.py` | Low/Medium | Yes, after repo-assist snapshots | repo-assist-eval, direct-template tests | Wrapper should not move retrieval logic first. |
| `repo-assist-eval` | `run_repo_assist_eval` | retrieval UX/eval | `cli/commands/repo_assist.py` | Low/Medium | Yes, after snapshots | repo-assist-eval 23/23 | Keep query-pack behavior stable. |
| `retrieval-index-build` | `run_retrieval_index_build` | retrieval | `cli/commands/retrieval.py` | Medium | Not first | retrieval MVP tests | Wrapper only; source-policy logic stays in retrieval module. |
| `retrieval-search` | `run_retrieval_search` | retrieval | `cli/commands/retrieval.py` | Medium | Not first | citation/search tests | Preserve local lexical-only scope. |
| `retrieval-citation-check` | `run_retrieval_citation_check` | retrieval | `cli/commands/retrieval.py` | Medium | Not first | citation validator tests | Do not weaken citation validation. |
| `retrieval-answer` | `run_retrieval_answer` | retrieval answer | `cli/commands/retrieval.py` | Medium | Not first | claim-support tests, abstention tests | Keep claim verifier behavior locked. |
| `hidden-eval-seed-run` | `run_hidden_eval_seed_run` | hidden eval | `cli/commands/hidden_eval.py` | Medium/High | No | hidden privacy tests, summary shape tests | Private target handling makes this sensitive. |
| `hidden-eval-rerun-failed` | `run_hidden_eval_rerun_failed` | hidden eval | `cli/commands/hidden_eval.py` | Medium/High | No | targeted rerun tests, privacy tests | Delta report must stay public-safe. |
| `trajectory-list` | `run_trajectory_list` | trajectory UX | `cli/commands/trajectory.py` | Low/Medium | Later | trajectory list/search tests | Query/display wrapper. |
| `trajectory-show` | `run_trajectory_show` | trajectory UX | `cli/commands/trajectory.py` | Low/Medium | Later | missing run-id validation, output shape | Argument validation currently central. |
| `trajectory-search` | `run_trajectory_search` | trajectory UX | `cli/commands/trajectory.py` | Low/Medium | Later | missing query validation, filter tests | Preserve filters. |
| `trajectory-export-sft` | `run_trajectory_export_sft` | trajectory export | `cli/commands/trajectory.py` | High | No | SFT-positive strictness tests | Export must not inflate eligibility. |
| `trajectory-export-preferences` | `run_trajectory_export_preferences` | trajectory export | `cli/commands/trajectory.py` | High | No | preference export tests | Safety-sensitive future-use output. |
| `trajectory-export-retrieval` | `run_trajectory_export_retrieval` | trajectory export | `cli/commands/trajectory.py` | High | No | retrieval export tests | Keep non-training scope clear. |
| `trajectory-quality-audit` | `run_trajectory_quality_audit` | trajectory quality | `cli/commands/trajectory_quality.py` | High | No | SFT-positive remains 0 test | Eligibility logic must remain untouched. |
| `agent-plan` | `run_agent_plan` | agent | `cli/commands/agent.py` | Medium | Later | agent plan report shape | Wrapper around orchestrator. |
| `agent-solve` | `run_agent_solve` | agent | `cli/commands/agent.py` | High | No | rollback/verifier tests, repair-loop tests | Verifier authority and rollback are critical. |
| `agent-verify` | `run_agent_verify` | agent | `cli/commands/agent.py` | Medium/High | No | verifier report tests | Keep verifier final authority. |
| `agent-backend-smoke` | `run_agent_backend_smoke` | backend smoke | `cli/commands/backend_smoke.py` | Medium/High | No | fail-closed, schema, normalization tests | Backend contract output must remain stable. |
| `agent-backend-provision-tiny-model` | `run_agent_backend_provision_tiny_model` | backend provisioning | `cli/commands/backend_smoke.py` | Medium/High | No | no-download mock tests, allowlist tests | Network/model writes require guardrails. |
| `agent-backend-provision-small-candidate` | `run_agent_backend_provision_small_candidate` | backend provisioning | `cli/commands/backend_smoke.py` | Medium/High | No | large-model guard tests | Do not weaken model guard. |
| `candidate-readiness-smoke` | `run_candidate_readiness_smoke` | model selection | `cli/commands/candidate_readiness.py` | Medium/High | No | candidate readiness tests | Pre-bakeoff signal only. |
| `tokenizer` | `run_tokenizer` | tokenizer/data | `cli/commands/data.py` | Medium | Later | tokenizer smoke tests | Writes tokenizer artifacts. |
| `download` | `run_download` | data download | `cli/commands/data.py` | High | No | safe download tests | Network/data fallback behavior is sensitive. |
| `download-safe` | `run_download_safe` | data download | `cli/commands/data.py` | High | No | safe defaults tests | Must not broaden source policy. |
| `download-core` | `run_download_core` | data download | `cli/commands/data.py` | High | No | core bootstrap tests | Data provenance sensitive. |
| `download-status` | `run_download_status` | data status | `cli/commands/data.py` | Low/Medium | Later | manifest summary tests | Display/status only but tied to data manifests. |
| `token-manifest` | `run_token_manifest` | data/token artifacts | `cli/commands/data.py` | Medium | Later | manifest reconstruction tests | Artifact metadata truthfulness. |
| `token-integrity` | `run_token_integrity` | data/token artifacts | `cli/commands/data.py` | Medium | Later | integrity report tests | Must keep no-legal-clearance limits. |
| `hardware-validate` | `run_hardware_validate` | hardware validation | `cli/commands/hardware.py` | Low/Medium | Later | hardware report shape tests | Current-machine evidence only. |
| `gpu-fit-validate` | `run_gpu_fit_validate` | hardware validation | `cli/commands/hardware.py` | Medium | Later | 4GB fit report tests | Do not overstate readiness. |
| `data-governance` | `run_data_governance` | governance | `cli/commands/governance.py` | High | No | governance blocker tests | Legal/source policy claims are sensitive. |
| `validate-real-path` | `run_validate_real_path` | validation | `cli/commands/validation.py` | Medium/High | No | validation report tests | Tiny smoke only, no model quality. |
| `validate-short-run` | `run_validate_short_run` | validation | `cli/commands/validation.py` | Medium/High | No | checkpoint/resume tests | Touches training-like flow, keep later. |
| `train-preflight` | `run_train_preflight` | training gate | `cli/commands/training.py` | High | No | preflight blocker tests | Training gate must remain strict. |
| `deployment-info` | `run_deployment_info` | deployment status | `cli/commands/deployment.py` | Low/Medium | Later | deployment truthfulness tests | Informational. |
| `train` | `run_train` | training | `cli/commands/training.py` | Very High | No | full training gate tests | Leave last. |
| `sft` | `run_sft` | training | `cli/commands/training.py` | Very High | No | SFT blocker tests | Leave last. |
| `dpo` | `run_dpo` | training | `cli/commands/training.py` | Very High | No | DPO blocker tests | Leave last. |
| `distill` | `run_distill` | training/distillation | `cli/commands/training.py` | Very High | No | distill blocker tests | Leave last. |
| `quantize` | `run_quantize` | quantization | `cli/commands/quantization.py` | High | No | fatal export tests | Preserve no-fake-success export behavior. |
| `hw-profile` | `run_hw_profile` | hardware config | `cli/commands/hardware.py` | Low/Medium | Later | output shape test | Display wrapper. |
| `eval-harness` | `run_eval_harness` | eval | `cli/commands/eval.py` | Medium/High | No | eval manifest tests | Keep external benchmark caveats. |
| `improve` | `run_improve` | self-improvement | `cli/commands/training.py` | Very High | No | no-auto-learning tests | Do not touch before Phase B decision. |
| `eval` | `run_eval` | eval | `cli/commands/eval.py` | Medium/High | No | eval report tests | Model-quality claims sensitive. |
| `benchmark-harness` | `run_benchmark_harness` | eval | `cli/commands/eval.py` | Medium/High | No | benchmark truthfulness tests | External comparison disabled/unverified. |
| `full` | `run_full` | pipeline orchestration | `cli/commands/full.py` | Very High | No | no-training-accident tests | Runs multi-stage pipeline; leave last. |

## 4. Current Boundary Problems

1. `run.py` mixes command parser declarations, dependency gates, command implementations, output formatting, report writing, and dispatch.
2. `main()` owns both parser construction and command dispatch. Future command additions increase merge conflict and regression risk.
3. Argument validation is partly central and partly inside handlers.
4. Output styles are repeated: text banners, JSON report path support, `quality_claim`, `private_leakage`, and safety summaries.
5. Command handlers import subsystem functions lazily, which is useful, but the wrapper logic has become large enough to need domain boundaries.
6. The safest safety scaffolding is in subsystem modules, not in a future CLI split; extraction should move wrappers first, not core logic.

## 5. Proposed Future CLI/Module Layout

Proposed layout for a later package only:

```text
cli/
  __init__.py
  parser.py
  dispatch.py
  output.py
  validation.py
  commands/
    status.py
    deps.py
    audit.py
    compile_source.py
    retrieval.py
    repo_assist.py
    hidden_eval.py
    trajectory.py
    trajectory_quality.py
    agent.py
    backend_smoke.py
    candidate_readiness.py
    data.py
    governance.py
    hardware.py
    validation_smoke.py
    deployment.py
    eval.py
    quantization.py
    training.py
    full.py
```

What stays in `run.py` later:

- Thin process entry point.
- `argparse` invocation delegated to `cli.parser`.
- Config application delegated to existing runtime config path.
- Dispatch call delegated to `cli.dispatch`.
- Stable `python run.py <command>` compatibility.

What should not be mixed:

- Hidden private target execution and public report rendering.
- Retrieval source-policy validation and CLI text formatting.
- Backend structured-output parsing and smoke CLI printing.
- Training commands and Phase A safety/reporting commands.
- Cloud planning docs and actual cloud execution.

## 6. Safe / Medium / High-Risk Extraction Groups

### Safe First Candidates

| Candidate | Why Safer | Files Involved Later | Tests Needed | Rollback Plan |
| --- | --- | --- | --- | --- |
| `status`, `deps`, `compile-source` wrappers | Mostly report/display code; no model/training/network path. | `run.py`, future `cli/commands/status.py`, `deps.py`, `compile_source.py` | command inventory snapshot, status/deps/compile-source output tests | Restore wrappers to `run.py`; command names unchanged. |
| Shared banner/output helpers | Repeated text formatting; behavior easy to snapshot. | `run.py`, future `cli/output.py` | help/output snapshot tests | Inline helpers back into `run.py`. |
| `repo-assist`/`repo-assist-eval` wrappers | Behavior is covered by `repo-assist-eval`; core retrieval stays put. | `run.py`, future `cli/commands/repo_assist.py` | repo-assist-eval 23/23, direct-template tests | Move wrapper code back. |
| Informational wrappers such as `deployment-info`, `hw-profile` | Mostly display/metadata. | `run.py`, future command modules | output shape tests | Move back. |

### Medium-Risk Candidates

| Candidate | Risk Reason | Tests Needed Before Movement |
| --- | --- | --- |
| Retrieval command wrappers | Citation/claim support must remain exact. | retrieval MVP tests, claim-support tests, out-of-scope abstention tests. |
| Hidden-eval wrappers | Privacy/reporting risk. | private leakage tests, targeted rerun tests, summary shape tests. |
| Backend/candidate smoke wrappers | Fail-closed/schema output risk. | backend normalization tests, candidate readiness tests, missing model tests. |
| Hardware/validation wrappers | Could overstate readiness if output changes. | hardware/gpu-fit/validation truthfulness snapshots. |

### High-Risk / Do Not Touch First

| Area | Why Protected |
| --- | --- |
| Training/preflight/SFT/DPO/distill/improve/full | Training readiness and Phase B truthfulness are sensitive. |
| `agent/backend.py` parser/normalization/path safety | Structured-output and safe-path policy must not weaken. |
| `agent/orchestrator.py` solve/repair/rollback/verifier flow | Verifier authority and rollback are critical. |
| `agent/trajectory_quality.py` | SFT-positive filtering must remain strict. |
| `eval_harness/hidden_seed_runner.py` private-target execution | Hidden/private leakage risk. |
| `retrieval/local_repo.py` source-policy/citation/claim verifier | Source-grounded truthfulness risk. |
| data governance/legal policy artifacts | No legal-clearance claim can be implied. |

## 7. Required Tests Before Extraction

Minimum tests before moving any command wrapper:

- Command inventory snapshot test: parser choices remain exactly stable.
- `python run.py --help` command-list stability test.
- Dispatch mapping snapshot test: command maps to expected handler group.
- JSON output key stability tests for high-signal commands.
- `repo-assist-eval` command test remains 23/23 with no web/model backend.
- Hidden-eval privacy test: `private_leakage: false`.
- Hidden-eval source/retrieval claim metrics remain present.
- Trajectory-quality audit test keeps SFT-positive at 0 unless real evidence says otherwise.
- Backend fail-closed smoke test for missing/unavailable backend.
- Candidate readiness unavailable/available tests with mocks.
- No training command accidentally run test.
- No cloud command accidentally run test.

Do not add extraction until these tests exist.

## 8. Staged Refactor Roadmap

### Pass 1: CLI Inventory and Snapshot Tests

Objective: freeze behavior before movement.

- Add command inventory snapshot test.
- Add `run.py --help` command-list stability test.
- Add dispatch table shape test.
- No handler movement.

Risk: low.

### Pass 2: Extract Pure Output / Report Helpers

Objective: reduce duplicate banner/report formatting only.

- Move banner printing and small output helpers to `cli/output.py`.
- Keep command logic in `run.py`.
- No command behavior change.

Risk: low/medium.

### Pass 3: Extract Low-Risk Local Command Wrappers

Objective: move lowest-risk wrappers while command names stay identical.

- Move `status`, `deps`, `compile-source`.
- Consider `deployment-info` and `hw-profile` after snapshots.
- Keep `python run.py <command>` stable.

Risk: low/medium.

### Pass 4: Extract Retrieval and Repo-Assist Command Group

Objective: move wrappers only, not source-policy logic.

- Move retrieval/repo-assist CLI wrappers.
- Keep `retrieval/local_repo.py` behavior unchanged.
- Require repo-assist/retrieval/claim-support tests.

Risk: medium.

### Pass 5: Extract Hidden-Eval Command Group

Objective: move hidden-eval CLI wrappers only.

- Move `hidden-eval-seed-run` and `hidden-eval-rerun-failed` wrappers.
- Keep private asset handling in `eval_harness/hidden_seed_runner.py`.
- Require private leakage and delta report tests.

Risk: medium/high.

### Pass 6: Extract Backend / Candidate Command Group

Objective: move backend smoke/provision/readiness wrappers.

- Keep parser, normalization, safe path, and model guard logic in `agent/backend.py` and `model_selection`.
- Require fail-closed, schema, missing-model, and large-model guard tests.

Risk: medium/high.

### Pass 7: Leave Training Commands Last

Objective: only after all safety tests and command snapshots are stable.

- Move training wrappers last, if ever.
- Do not change training readiness gates.
- Do not start Phase B.

Risk: very high.

## 9. Things Not To Touch

- `agent/trajectory_quality.py`
- `evals/hidden/private/hidden_eval_private_targets_v1.json`
- Hidden private target content and hidden-answer handling.
- `agent/backend.py` output normalization, schema validation, and safe-path rules.
- `agent/orchestrator.py` rollback/verifier authority.
- `retrieval/local_repo.py` source-policy, citation, and claim-support verifier behavior.
- `eval_harness/hidden_seed_runner.py` private/public boundary until snapshot tests exist.
- Training commands and training data.
- Cloud scripts and cloud plan guardrails.
- `.gitignore` generated/model artifact protections.

## 10. Verification Results

Verification commands for this planning pass:

- `python run.py status`
- `python run.py deps`
- `python run.py compile-source`
- `python run.py repo-assist-eval`
- `python run.py hidden-eval-seed-run`
- `python run.py trajectory-quality-audit`
- `python -m pytest tests -q`
- `python -m compileall -q .`
- `git diff --check`
- `git status --short`

Results:

| Command | Result | Notes |
| --- | --- | --- |
| `python run.py status` | Passed | Tokenizer/data cache present; pretrain/SFT/DPO/improved/eval artifacts missing as expected. |
| `python run.py deps` | Passed | Core deps 7/0; optional backend deps 2/0. |
| `python run.py compile-source` | Passed | `scanned: 103`, tracked-source-only scope. |
| `python run.py repo-assist-eval` | Passed | `23/23`, `uses_web: false`, `uses_model_backend: false`. |
| `python run.py hidden-eval-seed-run` | Passed | 17 attempted, 13 passed, 4 blocked, `private_leakage: false`, `quality_claim: none`. |
| `python run.py trajectory-quality-audit` | Passed | `scanned: 1370`, `sft_positive: 0`, `rejected_positive: 1370`. |
| `python -m pytest tests -q` | Passed | 237 passed, 1 skipped. |
| `python -m compileall -q .` | Passed | No output. |
| `git diff --check` | Passed | No whitespace errors. |
| `git status --short` | Passed | Only report artifacts are untracked. |

## 11. Next Recommended Package

Next package: `CLI command inventory/snapshot tests`.

Scope:

- Add a command inventory snapshot test from `run.py --help` or parser choices.
- Add a dispatch mapping snapshot test.
- Add no behavior changes and no handler movement.
- Run the same full verification set.

Do not start actual extraction until the command surface is locked by tests.
