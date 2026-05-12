# MY_LLM Repository System Map (Deep Technical Manifest)

## 0. Living Update Policy

- This file is the canonical architecture and operations manifest for this repository.
- Update this file when any of the following change:
  - Python modules are added, removed, renamed, or moved.
  - Tensor math, routing, caching, masking, or loss composition changes.
  - Config semantics, schema constraints, or override behavior changes.
  - Hardware profile fields, backend policy, AMP behavior, or runtime defaults change.
  - Artifact locations, telemetry contracts, or checkpoint/security policies change.
  - CLI command surface in `run.py` changes.

### Last synchronization

- Date: 2026-05-12
- Scope: Inventory, command-surface, agent-subsystem synchronization, Phase A hidden-eval/governance artifacts, Phase A base-model bakeoff protocol artifacts, the Phase A backend integration planning artifacts, and the Phase A retrieval-design planning artifacts. This was not a brand-new full-model audit.
- Module count: 93 Python modules.
- Runtime verification:
  - Default production-profile parameter cardinality is unverified in this pass.
  - The previously recorded 439,613,216 parameter count applies to the legacy experimental configuration, not the current default.

### Pass 1 truthfulness update

- Date: 2026-04-19
- Scope: CLI status/audit truthfulness, serving stub removal, quantization report integrity, and related tests only.
- `status` uses ASCII-safe filesystem status rows.
- `audit` is intentionally narrow for this pass: compile check, tokenizer smoke, status command health, fake serving detection, and quantization report integrity.
- `serving/server.py` no longer returns echo responses; it fails closed until a real model backend is configured.
- `quant_utils.py` supports CPU dynamic int8 only. Requested TorchScript export failures are fatal and cannot be recorded as successful quantization.
- Training quality, benchmark quality, dataset integrity, and production readiness remain unverified by this pass.
- `status` must route training through `train-preflight`; direct `train` recommendation before preflight is misleading.
- `token-manifest` reconstructs local token artifact metadata from `.bin` files only. It does not reconstruct licensing, filtering, deduplication, shuffle, or contamination provenance.
- `deps`, `token-integrity`, and `validate-real-path` provide machine-readable readiness evidence. `validate-real-path` performs a tiny real-token smoke path only and makes no model-quality claim.

### Hard-evidence blocker update

- Date: 2026-04-19
- Scope: hardware-readiness reporting, data-governance evidence, bounded exact dedup evidence, source-level benchmark-risk reporting, and short multi-step real-token validation.
- `hardware-validate` reports current-machine CUDA/device/dtype evidence and runs only the strongest safe narrow validation available. It distinguishes system GPU visibility through `nvidia-smi` from actual PyTorch CUDA usability; CPU-only or system-GPU-only evidence does not prove GPU readiness.
- `gpu-fit-validate` reports a bounded RTX 2050/4GB constrained-fit matrix through the repo-local `.venv` interpreter. Passing reduced configs classifies the hardware as validation-only for the current repo target, not pretraining-ready.
- `data-governance` emits source registry/category evidence, license-status evidence, exact token-chunk duplicate evidence with explicit scan coverage, bounded train/validation exact-overlap evidence, and source-level benchmark-risk evidence. It does not claim legal clearance, near-deduplication, semantic deduplication, or content contamination detection.
- `data_governance/source_license_metadata.json` is a manual repo-side source governance layer. It documents current review status, repo policy status, and training blocker level only; it is not legal advice and not legal clearance. Most declared licenses remain `unknown`; a tiny manually reviewed subset now has documented license metadata without changing `legal_clearance_claim`.
- `validate-short-run` runs a bounded multi-step real-token train/resume/checkpoint/eval-gate validation path. It is not pretraining and makes no model-quality claim.
- Real pretraining readiness remains controlled by `train-preflight`; validation reports do not override preflight.

### Agent Phase 1 update

- Date: 2026-05-03
- Scope: Added a new adjacent `agent/` subsystem for a minimum viable verified coding-agent loop.
- New CLI commands:
  - `agent-plan`
  - `agent-solve`
  - `agent-verify`
- Phase 1 supports deterministic exact-task routing, conservative coding-task routing, targeted context gathering, machine-readable planning, fail-closed backend loading, reversible workspace edits, structured verification, and machine-readable agent reports.
- Phase 1 does not include a repair loop, optimizer loop, trajectory learning, factual retrieval subsystem, or open-ended generation backend.
- Exact symbolic tasks are hard-routed to deterministic handlers or return unsupported; they do not fall back to generation.
- Coding tasks fail closed when no backend is configured or when no meaningful validator exists.

### Agent Phase 2 update

- Date: 2026-05-03
- Scope: Added an evidence-driven critic, bounded repair loop, post-green optimizer, and structured attempt history to the `agent/` subsystem.
- Coding tasks can now perform a bounded initial-plus-repair sequence when a backend supplies a repair candidate. Every failed attempt is verified, critiqued, rolled back, and recorded before the next repair is attempted.
- The optimizer runs only after a verified-green result and reruns the same verification suite. Regressing optimization attempts are rolled back and reported explicitly.
- Exact symbolic tasks still bypass the coding repair loop and remain deterministic-only.
- Phase 2 still does not include trajectory retrieval, learning/export hooks, broad factual retrieval, or production model-backend integration. Those remain deferred.

### Agent Phase 3 update

- Date: 2026-05-03
- Scope: Added replayable trajectory storage, deterministic metadata-based retrieval, and thin trajectory CLI surfaces to the `agent/` subsystem.
- Agent runs now write machine-readable trajectories under `run_artifacts/agent_trajectories/<run_id>/trajectory.json`.
- `trajectory-list`, `trajectory-show`, and `trajectory-search` expose stored trajectory metadata without claiming learning, semantic memory, or factual retrieval.
- Retrieval is conservative and deterministic: metadata filters plus simple lexical ranking. It is useful for replay and small context hints, not a semantic memory system.
- Phase 3 stores and retrieves prior experience only. Exporters, training hooks, preference data generation, and semantic memory remain explicitly deferred to Phase 4.

### Agent Phase 4 update

- Date: 2026-05-03
- Scope: Added strict trajectory export hooks for future supervised, preference, and retrieval workflows.
- New export commands:
  - `trajectory-export-sft`
  - `trajectory-export-preferences`
  - `trajectory-export-retrieval`
- Export artifacts are written under `run_artifacts/agent_exports/<export_run_id>/`.
- Export record families are versioned:
  - `agent_sft_export_v1`
  - `agent_preference_export_v1`
  - `agent_retrieval_export_v1`
- Export filtering is strict:
  - only `verified_success` solve trajectories feed positive SFT export
  - failed or blocked trajectories never become positive SFT demonstrations
  - preference pairs are emitted only when verifier evidence supports winner-over-loser ranking
  - retrieval exports may include verified, failed, blocked, unsupported, and planning-only trajectories, but each is labeled explicitly by outcome type
- Every exported record carries provenance, schema versioning, and `learning_claim: none`.
- Phase 4 creates learning hooks only. It does not retrain the model, update weights, add semantic memory, or prove model improvement.

### Phase A hidden eval backbone update

- Date: 2026-05-03
- Scope: Added the first private hidden-eval backbone artifact set under `evals/hidden/`.
- `evals/hidden/hidden_eval_spec_v1.md` and `hidden_eval_spec_v1.json` define the decision-oriented private categories for exact symbolic correctness, abstention/tool-or-source choice, source-grounded truthfulness, coding patch success, coding repair-loop success, retrieval-grounded QA, and verifier-loop KPI auditing.
- `evals/hidden/hidden_eval_seed_set_v1.jsonl` is a small private seed set with explicit frozen vs rotating split metadata, scoring-rule references, required evidence type, and training exclusion flags.
- `evals/hidden/private/hidden_eval_private_targets_v1.json` stores hidden targets and behavior assertions for private evaluation only; it is not user-facing help text.
- `evals/hidden/hidden_eval_holdout_hash_registry_v1.json` tracks seed-item hashes, private-target hashes, referenced artifact hashes, split status, and category counts for auditability.
- `eval_harness/hidden_eval_assets.py` provides narrow integrity loading and validation only. It does not run a bakeoff, score models, or redesign the broader eval harness.

### Phase A governance training-readiness update

- Date: 2026-05-03
- Scope: Added a manual source review template, a tiny repo-policy-allowed subset manifest, and an explicit benchmark holdout policy for future post-training readiness.
- `data_governance/source_review_template_v1.md` separates documented metadata, repo policy, training-readiness intent, and legal-safety limits.
- `data_governance/allowed_corpus_manifest_v1.json` is intentionally tiny and currently permits only a narrow repo-policy allowance where the repo has documented evidence strong enough to do so honestly.
- `data_governance/benchmark_holdout_policy_v1.md` ties configured benchmark exclusions and hidden-eval assets to training exclusion rules.
- This package does not provide legal clearance. `legal_clearance_claim` remains `none`, and most configured sources remain blocked or excluded.

### Phase A base-model bakeoff protocol update

- Date: 2026-05-03
- Scope: Added a narrow base-model decision framework under `model_selection/` for future Phase A bakeoff execution.
- `model_selection/bakeoff_protocol_v1.md` and `bakeoff_protocol_v1.json` define scope, hardware assumptions, required hidden-eval inputs, decision rules, and what does not belong in the bakeoff.
- `model_selection/bakeoff_candidate_matrix_v1.json` defines exactly three candidate slots: current custom base, one open dense 7B-class slot, and one open dense 14B-class slot.
- The current custom base is explicitly not a full contender unless a real usable checkpoint exists. As of protocol v1, the standard repo checkpoint/status surface does not evidence such a checkpoint.
- `model_selection/bakeoff_scoring_v1.md` and `bakeoff_scoring_v1.json` prioritize hidden verified task success, truthfulness, abstention, runtime fit, structured output reliability, and cost per verified success over public benchmark prestige.

### Phase A backend integration planning update

- Date: 2026-05-03
- Scope: Added a narrow first-backend integration plan under `backend_integration/`.
- The chosen first backend path is intentionally conservative: one Linux-first in-process local Transformers backend using the existing `Candidate` contract rather than a broader service-style or multi-backend design.
- `backend_integration/backend_integration_plan_v1.*` defines the one-path-first decision, touchpoints, and explicit deferrals.
- `backend_integration/backend_capability_gate_v1.*` defines the minimum integration bar: structured candidate validity, parse reliability, usefulness over the no-backend baseline, exact-task bypass preservation, verifier compatibility, retry-loop compatibility, and workstation runtime stability.
- `backend_integration/backend_failure_contract_v1.*` defines explicit backend failure classes and keeps backend failures from being misreported as successful solves.
- This package does not integrate a backend. It defines the first honest path and its gates only.

### Phase A retrieval design planning update

- Date: 2026-05-12
- Scope: Added a narrow first-phase retrieval design package under `retrieval_design/`.
- The chosen first retrieval scope is intentionally narrower than a generic RAG story: repo-authored local documentation plus repo-generated local manifests and reports only.
- External dataset content, downloaded corpora, benchmark exclusions, and all hidden-eval artifacts remain out of the first retrieval scope.
- `retrieval_design/retrieval_design_v1.*` defines the scope, non-goals, and explicit deferrals.
- `retrieval_design/retrieval_source_policy_v1.*` defines source classes and makes clear that repo-policy training allowance does not imply retrieval approval.
- `retrieval_design/citation_contract_v1.*` defines a checkable citation structure and states that citations are evidence references, not proof by themselves.
- `retrieval_design/retrieval_abstention_policy_v1.md`, `retrieval_design/retrieval_verifier_contract_v1.*`, and `retrieval_design/retrieval_benchmark_exclusion_policy_v1.md` define fallback behavior, verifier authority, and hidden/benchmark exclusion handling.
- This package does not implement retrieval, vector search, web search, or memory.

---

## 1. Crawl Scope and Ground Truth

### 1.1 Scanned locations

- Repository root (`*.py`)
- `agent/`
- `core/`
- `engine/`
- `eval/`
- `eval_harness/`
- `observability/`
- `safety/`
- `security/`
- `serving/`
- `tests/`

### 1.2 Excluded from code crawl

- `.venv/` third-party packages
- Binary/token cache and checkpoint payloads

### 1.3 Python module inventory

#### Root (15)

- `alignment.py`
- `config.py`
- `data.py`
- `distillation_trainer.py`
- `download_data.py`
- `eval_suite.py`
- `hardware_profiles.py`
- `infinite_improver.py`
- `model.py`
- `quant_utils.py`
- `run.py`
- `sft_trainer.py`
- `tokenizer.py`
- `train.py`
- `train_tokenizer.py`

#### agent/ (15)

- `agent/__init__.py`
- `agent/backend.py`
- `agent/context.py`
- `agent/critic.py`
- `agent/exact_tools.py`
- `agent/exporters.py`
- `agent/optimize.py`
- `agent/orchestrator.py`
- `agent/planner.py`
- `agent/retrieval.py`
- `agent/router.py`
- `agent/trajectory.py`
- `agent/types.py`
- `agent/verify.py`
- `agent/workspace.py`

#### core/ (11)

- `core/__init__.py`
- `core/checkpoint_io.py`
- `core/config_manager.py`
- `core/config_schema.py`
- `core/dependency_checks.py`
- `core/hardware_optim.py`
- `core/logging.py`
- `core/lr_schedule.py`
- `core/ops.py`
- `core/sequence_ops.py`
- `core/training_lifecycle.py`

#### engine/ (2)

- `engine/__init__.py`
- `engine/optimizer_factory.py`

#### eval/ (2)

- `eval/__init__.py`
- `eval/benchmark_harness.py`

#### eval_harness/ (6)

- `eval_harness/__init__.py`
- `eval_harness/contamination_checks.py`
- `eval_harness/hidden_eval_assets.py`
- `eval_harness/manifests.py`
- `eval_harness/runner.py`
- `eval_harness/statistical_tests.py`

#### observability/ (6)

- `observability/__init__.py`
- `observability/metrics.py`
- `observability/run_registry.py`
- `observability/security_events.py`
- `observability/tracing.py`
- `observability/training_telemetry.py`

#### safety/ (3)

- `safety/__init__.py`
- `safety/policy_filters.py`
- `safety/prompt_attack_checks.py`

#### security/ (2)

- `security/__init__.py`
- `security/validator.py`

#### serving/ (1)

- `serving/server.py`

#### tests/ (22)

- `tests/test_agent_phase1.py`
- `tests/test_agent_phase2.py`
- `tests/test_agent_phase3.py`
- `tests/test_agent_phase4.py`
- `tests/test_audit_truthfulness.py`
- `tests/test_bakeoff_protocol_assets.py`
- `tests/test_backend_integration_plan_assets.py`
- `tests/test_cli_truthfulness.py`
- `tests/test_config_profiles.py`
- `tests/test_core_lr_schedule.py`
- `tests/test_core_sequence_ops.py`
- `tests/test_data_token_truthfulness.py`
- `tests/test_deployment_profiles.py`
- `tests/test_eval_integrity.py`
- `tests/test_hidden_eval_assets.py`
- `tests/test_hard_evidence.py`
- `tests/test_kv_cache.py`
- `tests/test_model_shapes.py`
- `tests/test_quantization_truthfulness.py`
- `tests/test_retrieval_design_assets.py`
- `tests/test_serving_truthfulness.py`
- `tests/test_training_safety.py`

---

## 2. System Purpose and Architectural Shape

This codebase is an end-to-end small language model stack with strict runtime governance and multiple post-pretrain phases built on one shared model core.

Primary architectural planes:

- Control plane:
  - Dataclass runtime state in `config.py`
  - Strict section validation in `core/config_schema.py`
  - Hierarchical file/env ingestion in `core/config_manager.py`
- Orchestration plane:
  - Command routing and staged workflow in `run.py`
- Agent plane:
  - Verified coding-agent subsystem in `agent/*`
  - Deterministic exact-task routing, bounded repair, trajectory storage, conservative retrieval, and strict export hooks
- Model plane:
  - Default dense decoder-only GQA transformer in `model.py`
  - Legacy experimental profile gates MoE, memory tokens, recursive loops, MoD routing, and confidence head
  - Attention kernels and QK-Norm in `core/ops.py`
- Data plane:
  - Download/cache materialization in `download_data.py`
  - Runtime dataset blending/filtering in `data.py`
- Runtime optimization plane:
  - AMP and optimizer policy in `engine/optimizer_factory.py`
  - Shared scheduling and step cadence in `core/lr_schedule.py`, `core/training_lifecycle.py`
- Evaluation plane:
  - Core evaluator in `eval_suite.py`
  - Benchmark exclusion coverage, manifests, and statistics in `eval_harness/*`
  - Report-card harness and disabled external-comparison warning outputs in `eval/benchmark_harness.py`
- Integrity and telemetry plane:
  - Checkpoint integrity in `core/checkpoint_io.py`
  - Input and execution policy in `security/validator.py`
  - Step/hardware/security telemetry in `observability/*`
- Serving/safety plane:
  - Fail-closed serving boundary in `serving/server.py`
  - Prompt/output filters in `safety/*`

---

## 3. Command Surface and Pipeline Routing (`run.py`)

### 3.1 Supported commands

- `agent-plan`
- `agent-solve`
- `agent-verify`
- `trajectory-list`
- `trajectory-show`
- `trajectory-search`
- `trajectory-export-sft`
- `trajectory-export-preferences`
- `trajectory-export-retrieval`
- `tokenizer`
- `download`
- `download-safe`
- `download-core`
- `download-status`
- `token-manifest`
- `token-integrity`
- `deps`
- `hardware-validate`
- `gpu-fit-validate`
- `data-governance`
- `validate-real-path`
- `validate-short-run`
- `train-preflight`
- `deployment-info`
- `train`
- `sft`
- `dpo`
- `distill`
- `quantize`
- `hw-profile`
- `eval-harness`
- `improve`
- `eval`
- `benchmark-harness`
- `audit`
- `full`
- `status`

### 3.2 Runtime config application before dispatch

`run.py` applies runtime config before command execution:

1. Load layered configs from `--config` and env vars.
2. Apply overrides into global config objects.
3. Emit snapshot artifacts under `run_artifacts/`:
   - `runtime_config_snapshot_<ts>.json`
   - `runtime_config_effective_<ts>.json`

### 3.3 Dependency gate behavior

- `check_dependencies()` validates required imports per command.
- Missing required packages cause immediate process exit with install hints.
- Optional `regex` package emits advisory tip only.

### 3.4 Built-in operational commands

- `status`: filesystem-based stage completion visibility
- `download-status`: download manifest summary and failure hotspots
- `token-manifest`: reconstructs a token artifact manifest from existing local token binaries with explicit reconstructed-provenance caveats
- `token-integrity`: checks local token artifact sizes, token counts, split pairing, and labeled full-or-sampled hashes without claiming data quality
- `deps`: checks the installed environment against the declared requirement file
- `hardware-validate`: reports current-machine hardware evidence and runs a narrow CPU or CUDA validation step without claiming full training fit
- `gpu-fit-validate`: runs a small bounded CUDA fit matrix on the repo-local `.venv` interpreter, records peak memory/checkpoint/resume evidence, and classifies 4GB RTX 2050 hardware as validation-only for the current target
- `data-governance`: writes source/category/license-status, manual source metadata coverage, repo policy status, training blocker level, exact dedup scan coverage, bounded train/validation exact-overlap, and source-level benchmark-risk evidence without claiming legal clearance, near-deduplication, or content contamination detection
- `validate-real-path`: runs a tiny real-token train/checkpoint/resume/eval-gate smoke path and writes a JSON report; it does not start pretraining or claim model quality
- `validate-short-run`: runs a bounded multi-step real-token validation with checkpoint reload/resume and continuation; it is not pretraining
- `train-preflight`: no-training readiness checks for declared dependency drift, config geometry, tokenizer loadability, local token cache/manifest presence, checkpoint directory expectations, dtype/device compatibility, and parameter-memory lower-bound reporting
- `deployment-info`: read-only deployment tier metadata and lower-bound memory estimates; does not validate runtime deployment
- `agent-plan`: writes a machine-readable planning report without claiming solve success
- `agent-solve`: runs the verified coding-agent solve path, including deterministic exact-task handling, bounded repair attempts, and optional post-green optimization
- `agent-verify`: runs agent-side verification only and does not fabricate a candidate
- `trajectory-list` / `trajectory-show` / `trajectory-search`: inspect stored coding-agent trajectories with deterministic metadata filters and compact summaries
- `trajectory-export-sft` / `trajectory-export-preferences` / `trajectory-export-retrieval`: export stored trajectories into future learning-use artifacts with strict provenance and filtering; these commands do not retrain the model
- `audit`: Pass 1 truthfulness checks covering compile, tokenizer smoke, status health, fake-serving detection, and quantization report integrity

---

## 4. Configuration and Validation Contract

### 4.1 Runtime source-of-truth objects (`config.py`)

Global singleton config objects:

- `model_cfg`
- `train_cfg`
- `sft_cfg`
- `dpo_cfg`
- `improver_cfg`
- `eval_cfg`
- `distill_cfg`
- `quant_cfg`
- `hardware_profile_cfg`

`validate_runtime_state()` is called at import time and validates all sections through `core.config_schema.validate_runtime_sections(...)`.

### 4.2 Key model defaults and derived fields

Important default `ModelConfig` production profile:

- `profile_name="edge_0_5b_gqa"`
- `dim=1536`, `n_layers=18`, `n_heads=16`, `n_kv_heads=4`
- `max_seq_len=2048`, `ffn_dim=4096`
- `use_moe=False`, `n_memory_tokens=0`, `n_loops=1`
- `yarn_scale=1.0`, `sliding_window=512`, `rope_theta=500000.0`
- `attention_backend="auto"`, `use_flashattention=False`
- `use_mla=False`, `mla_compression_ratio=0.5`
- `qk_norm_eps=1e-6`, `use_confidence_head=False`, `use_mod_routing=False`
- Explicit profiles: `edge_0_5b_gqa`, `local_1_5b_gqa`, `workstation_3b_gqa`, `legacy_experimental_439m`

Derived and validated in `__post_init__`:

- `head_dim = dim // n_heads`
- `latent_head_dim = int(head_dim * mla_compression_ratio)`
- Backend must be one of `{sdpa, flash2, auto}`
- MLA ratio must be `0.25` or `0.5`
- `qk_norm_eps > 0`

### 4.2.1 Deployment tier metadata

`config.py` defines deployment metadata tiers:

- `2gb_edge`
  - Intended as edge/offload territory, not a claim that 3B/5B GPU inference fits in 2GB VRAM.
  - Uses `edge_0_5b_gqa` as the nearest in-repo model profile reference.
  - Assumes small active context and RAG for document workflows.
- `8gb_laptop`
  - Uses `edge_0_5b_gqa` as the conservative reference profile.
  - Larger local models require quantization/offload paths not verified by this repo.
- `16gb_workstation`
  - Uses `local_1_5b_gqa` as the reference profile.
  - `workstation_3b_gqa` is not claimed to fit with useful context without measured memory checks.

Deployment helper truth limits:

- `build_deployment_report(...)` reports lower-bound estimates only.
- Weight estimates exclude optimizer state, gradients, activations, allocator overhead, framework overhead, and fragmentation.
- KV-cache estimates exclude allocator/framework overhead and still grow with context in global layers.
- PyTorch CPU/GPU is the current in-repo reference path.
- Dynamic-int8 CPU quantization is implemented in `quant_utils.py`; GGUF, GPTQ, AWQ, vLLM, and llama.cpp are external/unverified targets only.
- A deployment tier is metadata, not validated runtime support, benchmark evidence, or proof of training quality.

### 4.3 Key pretrain defaults (`TrainConfig`)

- Token budget and step geometry:
  - `total_tokens=10_000_000_000`
  - `batch_size=64`, `seq_len=512`, `grad_accum=4`
- LR schedule:
  - `max_lr=3e-4`, `min_lr=3e-5`, `warmup_tokens=200_000_000`
- AMP and compile:
  - `dtype="bfloat16"`, `amp_grad_scaler=True`, `compile_model=True`
- Data runtime knobs:
  - `data_num_workers=0`, `data_prefetch_queue=8`
  - `data_prefetch_factor=2`, `data_pin_memory=True`
  - `device_non_blocking=True`
- Loss weights:
  - `aux_loss_coeff=0.01`, `gate_loss_coeff=0.01`
  - `z_loss_coeff=0.001`, `unc_loss_coeff=0.1`
- Data quality:
  - `dedup_ngram_size=13`, `dedup_threshold=0.8` are retained config fields, not currently wired into runtime deduplication
  - `min_doc_length=50`, `max_doc_length=100000`
  - `quality_threshold=0.3` is retained config, not currently enforced by `data.py`

### 4.4 Override ingress and rollback semantics

- File/env ingestion:
  - `core/config_manager.ConfigManager.load(...)`
  - Supports JSON/YAML + env overrides (`MYLLM__SECTION__FIELD`)
- Apply semantics:
  - `apply_overrides(overrides, strict=False)` mutates singleton dataclasses
  - Re-runs section `__post_init__` methods
  - Re-validates full runtime state
  - Rolls back to full snapshot on any exception
- Strict mode:
  - unknown sections/fields raise `KeyError`

### 4.5 Schema-level hard constraints (`core/config_schema.py`)

Examples of enforced constraints:

- `model.n_heads % model.n_kv_heads == 0`
- `model.n_experts_active <= model.n_experts`
- `train.min_lr <= train.max_lr`
- `train.warmup_tokens <= train.total_tokens`
- `dpo.loss_type in {sigmoid, hinge}`
- `distill.alpha_kd + distill.alpha_ce > 0`

---

## 5. Canonical Pretrain Execution Path

### Stage A: Data loading and sample materialization

`get_dataloader(tokenizer, train_cfg, start_step)` creates:

1. `MixedTokenDataset`
2. `torch.utils.data.DataLoader`
3. `PrefetchDataLoader` wrapper

Each yielded sample:

- `input_ids`: length `seq_len`
- `targets`: next-token shifted labels

### Stage B: Forward pass entry (`train.py`)

Microbatch:

1. Move tensors to device (optional non-blocking)
2. Run under AMP autocast
3. `out = model(ids, targets=tgt)`

### Stage C: Core model pass (`model.py`)

1. Token embedding. Memory-token prepend is legacy-only.
2. `_run_layers(...)`:
    - Shallow layers once
   - Deep layers looped `n_loops` times only in the legacy experimental profile
   - Deep train path uses checkpointing (`use_reentrant=False`)
3. Block internals:
   - `RMSNorm -> GQAttention -> residual`
   - `RMSNorm -> SwiGLU -> residual` in production profiles
   - `MoE` and `MoDRouter` are legacy experimental mechanisms
4. Head path:
   - `RMSNorm -> tied LM head -> logits`
5. Training return:
   - `LossOutput(total_loss, ce_loss, aux_loss, load_balance_loss, z_loss, unc_loss)`

### Stage D: Train-time loss assembly

Optimizer scalar in `train.py`:

- `ce_loss`
- `+ aux_loss_coeff * aux_loss`
- `+ gate_loss_coeff * load_balance_loss`
- `+ z_loss_coeff * z_loss`
- `+ unc_loss_coeff * unc_loss`

Then divided by `grad_accum` before backward.

### Stage E: Step, validation, telemetry, checkpoint

1. Sync gate via `should_sync_step`
2. Grad clipping via AMP policy
3. Token-aware cosine LR schedule
4. Optimizer step + scaler update + zero-grad
5. Telemetry emit (`TrainingTelemetry`, `PerformanceMonitor`)
6. Validation cadence and early stopping
7. Checkpoint and best-checkpoint writes

---

## 6. Model Plane Deep Dive (`model.py`, `core/ops.py`)

### 6.1 Attention math and ordering

Inside `GQAttention.forward`:

1. Project Q/K/V
2. Optional MLA latent compression path (`k_down/v_down -> k_up/v_up`)
3. Apply `apply_qk_norm(q, k, eps=qk_norm_eps)`
4. Apply YaRN rotary embedding with offset
5. Append KV cache when provided
6. Expand grouped K/V to full query-head count
7. Dispatch kernel through `run_attention(...)`

### 6.2 Kernel dispatch policy (`core/ops.py`)

Flash path (`flash_attn_func`) is used only if all conditions pass:

- backend policy in `{auto, flash2}`
- `use_flashattention=True`
- non-local attention
- no custom attention mask
- flash bindings importable
- CUDA tensor
- dtype in `{float16, bfloat16}`
- `head_dim % 8 == 0`

Else fallback to SDPA (`torch.scaled_dot_product_attention`).

### 6.3 YaRN implementation details

- Extended rope cache length: `max_seq_len * yarn_scale`
- Per-dimension correction ramp between `beta_fast` and `beta_slow`
- Multiplicative scale factor for long-context compensation

### 6.4 Legacy MoE details

`MoE` exists only for the explicit `legacy_experimental_439m` profile:

- Router logits in fp32
- Top-k (`n_experts_active`) dispatch
- Switch-style load-balance term:
  - `n_experts * sum(load_fraction_i * prob_mean_i)`
- z-loss term:
  - `mean(logsumexp(router_logits)^2)` in fp32
- Grouped routing implementation:
  - flatten token-expert assignments
  - sort by expert
  - run expert MLP per expert span
  - `scatter_add_` weighted accumulation to token outputs

### 6.5 Legacy recursive layer schedule

- `n_shallow = n_layers // 2`
- Shallow half runs once
- Production profiles use `n_loops=1`
- Legacy experimental profile can run the deep half multiple times
- During training, deep layers are wrapped by checkpoint calls to cap activation memory

### 6.6 Legacy confidence head and adaptive refinement

When enabled by the explicit legacy profile:

- Confidence head predicts 3-channel uncertainty signals.
- Inference may trigger refinement when average confidence `< adaptive_loop_threshold`.
- Two refinement paths:
  - full recompute with cache rebuild (`_run_refinement_with_cache`)
  - cached single-token deep re-run (`_run_refinement_cached`)

### 6.7 Generation API behavior

`LLM.generate(...)`:

- Prefill pass obtains logits + KV cache
- Decode loop uses cached single-token forward with offset increments
- Supports temperature, top-k, top-p sampling
- Returns generated token ids + optional uncertainty tensor

---

## 7. Pretrain Runtime and Distributed Behavior (`train.py`)

### 7.0 Training preflight

`python run.py train-preflight` calls `run_training_preflight(...)` and does not start real training.

Implemented checks:

- Model/train config geometry, including `train.seq_len <= model.max_seq_len`
- Tokenizer artifact presence and loadability
- Local `.bin` token cache presence
- Token artifact manifest loadability and existing-artifact reporting
- Checkpoint directory state or parent-path expectation
- AMP dtype/device compatibility
- Dense production parameter-memory lower bound

Truth limits:

- A passed preflight does not prove benchmark quality, convergence, data quality, throughput, distributed correctness, or production safety.
- Memory reporting is a parameter lower bound only; optimizer state, gradients, activations, compile overhead, dataloader memory, and fragmentation are not included.
- Missing local token artifacts fail readiness because streaming dataset access is not verified by this preflight.

### 7.1 Startup sequence

1. Initialize distributed context (`setup()`)
2. Configure logging
3. Apply runtime config overlays (`_apply_runtime_config`)
4. Apply hardware profile attention policy (`_apply_attention_runtime_policy`)
5. Build telemetry stream path and logger
6. Build AMP policy and optimizer
7. Load tokenizer and checkpoint
8. Construct data loaders

### 7.2 Distributed specifics

- DDP enabled when `RANK` and `WORLD_SIZE` env vars exist.
- Backend selection:
  - CUDA + non-Windows -> `nccl`
  - otherwise -> `gloo`
- DDP wrapper options:
  - `find_unused_parameters=True`
  - `gradient_as_bucket_view=True`

### 7.3 Token-step math

- Tokens per optimizer step:
  - `batch_size * seq_len * grad_accum * world_size`
- Total planned steps:
  - `total_tokens // tokens_per_step`

### 7.4 OOM resilience

- Detects CUDA OOM via exception text.
- Skips microbatch and clears cache.
- Fails hard after >3 consecutive OOMs (`Fatal OOM loop detected`).
- Non-finite loss aborts training with `FloatingPointError`; it is not skipped as a recoverable batch.

### 7.5 Early stopping behavior

`EarlyStopping` defaults:

- `patience=20`
- `min_delta=0.001`
- `warmup=200` steps

Uses validation loss, not training loss, and writes `loss_history.json`.

### 7.6 Checkpoint policy

- Atomic checkpoint writes through `atomic_torch_save`.
- Rolling retention: keep latest 3 non-best checkpoints.
- Supports resume from latest checkpoint in directory.
- Best checkpoints are tagged with `_best`.

---

## 8. Data Plane Deep Dive (`data.py`, `download_data.py`)

### 8.1 Runtime source catalog (ALL_SOURCES)

Data sources include web, wiki, math, code, and instruction corpora. Current source IDs:

- `HuggingFaceFW/fineweb-edu`
- `wikimedia/wikipedia`
- `HuggingFaceTB/smollm-corpus`
- `open-phi/textbooks`
- `HuggingFaceTB/finemath`
- `open-web-math/open-web-math`
- `openai/gsm8k`
- `microsoft/orca-math-word-problems-200k`
- `TIGER-Lab/MathInstruct`
- `lighteval/MATH-Hard`
- `m-a-p/CodeFeedback-Filtered-Instruction`
- `ise-uiuc/Magicoder-Evol-Instruct-110K`
- `iamtarun/python_code_instructions_18k_alpaca`
- `codeparrot/github-code`
- `ajibawa-2023/Code-290k-ShareGPT`
- `bigcode/self-oss-instruct-sc2-exec-filter-50k`
- `nickrosh/Evol-Instruct-Code-80k-v1`
- `deepmind/code_contests`
- `code-search-net/code_search_net`
- `b-mc2/sql-create-context`
- `argilla/magpie-ultra-v0.1`
- `HuggingFaceH4/ultrachat_200k`
- `teknium/OpenHermes-2.5`
- `Open-Orca/SlimOrca`
- `google/boolq`
- `truthful_qa`
- `Anthropic/hh-rlhf`
- `allenai/ai2_arc`
- `winogrande`

Source activation excludes any dataset listed in `train_cfg.excluded_benchmark_sources` and optional downloader skip envs.

### 8.2 Implemented data filtering and truth limits

There is no implemented `DataQualityFilter` in the current runtime path.
The current data behavior is narrower:

1. Source-level exclusion via `train_cfg.excluded_benchmark_sources`
2. Optional downloader source skipping through environment variables
3. Text extraction with fallback fields
4. Empty/very short extracted text skipped by extractor checks
5. EOS append and token-cache materialization

Current code does not implement MinHash, near-duplicate detection, PII removal,
legal license clearance, or content-level benchmark contamination detection.
`data-governance` classifies configured source IDs and reports unknown licenses
truthfully; many active sources remain `needs_manual_review`.

### 8.2.1 Manual source-license metadata

- `data_governance/source_license_metadata.json` contains one record per configured source.
- The metadata tracks source category/domain, declared license string when manually documented, license evidence source, review basis, benchmark-risk status, governance classification, repo policy status, training blocker level, intended training role, notes, and `legal_clearance_claim`.
- `legal_clearance_claim` must remain `none`; `allowed_by_repo_policy` is not legal approval.
- Current metadata is complete for source coverage but still highly restrictive for training readiness. Most declared licenses remain `unknown`, and only a tiny manually reviewed subset has documented license metadata.
- Current triage remains intentionally conservative: most active sources stay `blocked_pending_review` and `hard_blocker`; configured benchmark sources are `excluded` and `hard_blocker`; any repo-policy allowance remains narrow and not legally cleared.

### 8.3 Mixed dataset runtime phases

`MixedTokenDataset.__iter__` has two phases:

- Disk phase:
  - Weighted sampling from cached `.bin` sources
  - Source exhaustion tracking and dynamic alive-set sampling
- Streaming phase:
  - Grouped source loading (`max_open_sources`)
  - Weighted inter-source sampling
  - Text extraction, tokenization, EOS append

### 8.4 Validation dataset behavior

`ValidationTokenDataset`:

- First preference: held-out binary files in `val_data_dir`
- Fallback: last 5% of first available training binary sources

### 8.5 Prefetch wrapper behavior

`PrefetchDataLoader`:

- Background thread prefetch queue
- Propagates worker exceptions by raising `DataPipelineError`
- Never silently swallows worker errors

### 8.6 Downloader and resume semantics

`download_data.py` features:

- Per-source token budgets from weighted allocation
- Partial resume from existing `.bin` files
- Validation split routing (`VAL_FRACTION`)
- Stream retry/load retry loops
- Disk-cap cutoff (`MAX_DISK_GB`)
- Manifest run history in `data_cache/download_manifest.json`
- Token artifact metadata in `data_cache/token_artifacts_manifest.json`
  - source/subset IDs
  - dataset split and artifact split
  - path existence
  - dtype/itemsize
  - size bytes and token counts
  - config hash for the downloader source plan

---

## 9. Phase-Specific Pipelines

### 9.1 Tokenizer training (`train_tokenizer.py`)

- Collects streaming text from curated sources
- Normalizes structured text differently from prose
- Trains byte-level BPE (`ByteLevelBPETokenizer`)
- Promotes code/math merges via bounded rank-shift policy
- Runs verification battery with fidelity and chars-per-token checks

### 9.2 Runtime tokenizer (`tokenizer.py`)

- Byte-level GPT-2 style tokenizer with special chat tokens
- Thread-safe LRU caches for BPE and domain encoding
- Regex-based splitting when `regex` package is available
- Longest-match trie for code/math domain path
- Chat templating and chat encode helpers
- Minimal audit helper reports vocab size, merge count, special-token integrity,
  round-trip smoke checks, and chars/token on caller-provided samples

### 9.3 SFT (`sft_trainer.py`)

- Loads JSON/JSONL conversation data with safety validators
- Masks loss to assistant tokens only
- Supports sequence packing
- Uses shared AMP, LR schedule, lifecycle helpers, telemetry, secure checkpoint I/O

### 9.4 DPO (`alignment.py`)

- Loads preference pairs (`prompt`, `chosen`, `rejected`)
- Builds frozen reference model via deep copy
- Computes masked sequence log-prob deltas
- Supports sigmoid and hinge DPO losses with optional label smoothing

### 9.5 Distillation (`distillation_trainer.py`)

- Teacher/student distillation from checkpoints
- Distill loss:
  - `alpha_kd * KL(student || teacher)`
  - `+ alpha_ce * CE`
  - `+ aux_loss_coeff * aux_loss`
  - `+ gate_loss_coeff * load_balance_loss`
- Strict recipe validation hard-fails if key hyperparameters differ from expected stable recipe.

### 9.6 Infinite improver (`infinite_improver.py`)

- Confidence-head based uncertainty is legacy-profile only; production-profile compatibility is unverified in this pass
- Generates diverse candidates with varied sampling temperatures
- Scores candidates by model self-likelihood, confidence where available, and heuristics
- Builds synthetic preference pairs
- Mixes anchor pairs from DPO data to reduce drift
- Runs iteration DPO and quality-gates by uncertainty regression

### 9.7 Evaluation suite (`eval_suite.py`)

- Requires a loadable checkpoint before writing an eval report; missing or unloadable checkpoints raise `EvalIntegrityError`
- Perplexity
- Multiple-choice QA
- GSM8K math
- Code generation + execution checks
- Consistency/self-contradiction proxy
- Confidence calibration (ECE)
- Operational stats and 18-metric scorecard output
- Report metadata includes checkpoint load status, checkpoint sidecar hash presence, model config hash, per-benchmark run modes, sample counts, and warnings
- Toy/fallback/smoke samples are labeled as smoke-only and are not real benchmark evidence

### 9.8 Eval harness (`eval_harness/*`)

- Wraps eval suite with run manifest
- Computes benchmark exclusion coverage from excluded-source overlap
- Does not implement content contamination detection
- Carries eval run mode, real-benchmark-evidence flag, and eval warnings into manifest outputs
- Emits compact phase marker (`phase_eval_harness.json`)

### 9.9 Benchmark harness (`eval/benchmark_harness.py`)

- Builds 18-metric report-card JSON
- External 1x/10x/100x scale-band comparison is disabled as unverified; the repo has no measured baseline evidence for those target bands
- Emits CSV and Markdown summaries with explicit disabled/unverified comparison warnings

### 9.10 Quantization (`quant_utils.py`)

- CPU dynamic int8 quantization of linear layers
- Optional TorchScript trace export; a requested TorchScript export failure raises `QuantizationExportError`
- Writes quant report JSON only through report validation
- ONNX export flag exists in config, but requesting it fails explicitly because no ONNX export path is implemented
- GGUF, GPTQ, AWQ, vLLM, and llama.cpp export/runtime paths are not implemented or verified in this repo

### 9.11 Serving (`serving/server.py`)

- Minimal fail-closed request handler
- Prompt-attack detection gate
- Output sanitization pass
- No default model backend and no echo response path

### 9.12 Local deployment honesty

- Current deployable reference path is PyTorch CPU/GPU execution using in-repo model code and checkpoints.
- `python run.py deployment-info` prints deployment tier metadata and lower-bound memory estimates.
- 2GB VRAM is explicitly treated as edge/offload territory.
- Long context materially increases KV-cache memory; local/sliding layers are bounded, but global layers keep full-context KV.
- Deployment readiness is not benchmark quality, training success, or production serving readiness.

---

## 10. Hardware Profile and Backend Resolution

### 10.1 Profile files

- `hardware_configs/t4_profile.json`
  - `preferred_dtype=float16`
  - `attention_backend=auto`
  - `use_flashattention=true`
- `hardware_configs/a10_profile.json`
  - `preferred_dtype=bfloat16`
  - `attention_backend=auto`
  - `use_flashattention=true`
- `hardware_configs/cpu_profile.json`
  - `preferred_dtype=float32`
  - `attention_backend=eager`
  - `use_flashattention=false`

### 10.2 Runtime resolution path

- Profile auto-detection by GPU name (`t4`, `a10`, else fallback behavior)
- Runtime resolver normalizes backend policy to `flash2` or `sdpa`
- Attention policy is applied at startup in:
  - `train.py`
  - `distillation_trainer.py`

---

## 11. Security, Safety, and Integrity Contracts

### 11.1 Input and path validation (`security/validator.py`)

- Root allowlists for local files
- Size-guarded JSON and JSONL readers
- Runtime production detection (`MYLLM_RUNTIME_ENV`, `MYLLM_ENV`, `ENV`)
- Sanitization helpers for bounded log/output text

### 11.2 Code execution policy

Code execution is disabled unless all conditions pass:

- Non-production runtime
- `MYLLM_ENABLE_CODE_EXEC=1`
- `MYLLM_CODE_EXEC_SCOPE` in `{offline_eval, benchmark, local_dev}`
- AST safety checks pass

Execution uses constrained subprocess mode (`python -I -S`) with timeout.

### 11.3 Checkpoint integrity (`core/checkpoint_io.py`)

- Atomic save semantics
- SHA256 sidecar generation and verification
- Optional/required JSON manifest verification
- `weights_only=True` load path
- Security events emitted on pass/fail outcomes

### 11.4 Safety filters (`safety/*`)

- Prompt attack detector via pattern matching
- Output policy redaction and blocklist checks

---

## 12. Observability and Telemetry Contract

### 12.1 Metrics primitives

- `MetricsLogger`: thread-safe JSONL append sink
- `trace_span`: context timing helper
- `RunRegistry`: lockfile-guarded run index

### 12.2 Security telemetry

- Structured security event stream with non-fatal emission policy
- Default path:
  - `run_artifacts/security_events.jsonl`

### 12.3 Training telemetry (`observability/training_telemetry.py`)

- Loss spike tracking with moving baseline ratio
- Gradient stats:
  - L2 norm
  - abs mean/max
  - zero fraction
- CUDA stats:
  - allocated/reserved/peak memory
  - optional `nvidia-smi` utilization, mem, temp
- Performance monitor:
  - instantaneous and EMA tokens/sec

---

## 13. Runtime Environment Variables

### 13.1 Config layering

- `MYLLM__...` (section-field overrides)
- `MYLLM_CONFIG`
- `MYLLM_CONFIG_PATHS`
- `MYLLM_DISABLE_ENV_OVERRIDES`
- `MYLLM_MAX_CONFIG_BYTES`
- `MYLLM_LOG_LEVEL`

### 13.2 Data and path safety

- `MYLLM_WORKSPACE_ROOT`
- `MYLLM_ALLOWED_DATA_ROOTS`
- `MYLLM_MAX_DATA_JSON_BYTES`
- `MYLLM_MAX_DATA_JSONL_LINE_BYTES`

### 13.3 Checkpoint integrity policy

- `MYLLM_ENFORCE_CHECKPOINT_SHA256`
- `MYLLM_CHECKPOINT_MANIFEST_REQUIRED`
- `MYLLM_CHECKPOINT_MANIFEST_PATH`
- `MYLLM_WRITE_CHECKPOINT_MANIFEST`

### 13.4 Telemetry and security events

- `MYLLM_SECURITY_EVENTS_ENABLED`
- `MYLLM_SECURITY_EVENTS_PATH`
- `MYLLM_LOSS_SPIKE_THRESHOLD`

### 13.5 Code execution policy

- `MYLLM_ENABLE_CODE_EXEC`
- `MYLLM_CODE_EXEC_SCOPE`
- `MYLLM_RUNTIME_ENV`
- `MYLLM_ENV`
- `ENV`

### 13.6 Downloader controls

- `DATA_CACHE_DIR`
- `DOWNLOAD_AUTO_PROCEED`
- `DOWNLOAD_NETWORK_SAFE`
- `DOWNLOAD_CORE_FIRST`
- `DOWNLOAD_SOURCE_LIMIT`
- `DOWNLOAD_SKIP_SOURCES`
- `DOWNLOAD_MAX_SOURCE_FAILURES`

---

## 14. Artifact Contracts and Directory Roles

### 14.1 Primary outputs

- Pretrain checkpoints: `checkpoints/`
- SFT checkpoints: `sft_checkpoints/`
- DPO checkpoints: `dpo_checkpoints/`
- Distill checkpoints: `distill_checkpoints/`
- Quantized outputs: `quantized/`
- Eval outputs: `eval_results/`

### 14.2 Runtime configuration artifacts

- `run_artifacts/runtime_config_snapshot_*.json`
- `run_artifacts/runtime_config_effective_*.json`
- `checkpoints/runtime_config_snapshot.json`
- `checkpoints/runtime_config_effective.json`

### 14.3 Telemetry and security artifacts

- Trainer phase telemetry JSONL under each output directory
- Security events JSONL stream (default under `run_artifacts/`)

### 14.4 Data cache artifacts

- `data_cache/tokens/*.bin`
- `data_cache/val_tokens/*.bin`
- `data_cache/download_manifest.json`

### 14.5 Eval harness and benchmark artifacts

- `eval_results/manifest.json`
- `eval_results/phase_eval_harness.json`
- `eval_results/benchmark_harness_report.json`
- `eval_results/benchmark_harness_report_card_18.json`
- `eval_results/benchmark_harness_scale_comparison.csv`
- `eval_results/benchmark_harness_summary.md`

---

## 15. File-by-File Technical Manifest (Entry-Point Focus)

### 15.1 Root modules

- `run.py`
  - Entry command router, dependency gate, runtime config snapshots, full-pipeline orchestrator
- `config.py`
  - Dataclass runtime state, deployment tier metadata/helpers, override application, rollback on validation failure
- `model.py`
  - Core model graph: RMSNorm, YaRN/RoPE cache, GQAttention, dense SwiGLU production path, legacy experimental modules, generation
- `train.py`
  - Pretrain loop, DDP/AMP integration, token-aware scheduler, telemetry, early stop, checkpointing
- `data.py`
  - Mixed source dataset, validation dataset, prefetch wrapper
- `download_data.py`
  - Source downloader, budget allocation, resume, run manifest, token-artifact manifest tracking
- `tokenizer.py`
  - Runtime BPE tokenizer with code/math path and chat helpers
- `train_tokenizer.py`
  - Corpus collection, tokenizer training, merge promotion, verification suite
- `sft_trainer.py`
  - Assistant-only loss SFT with packing and telemetry
- `alignment.py`
  - DPO training with frozen reference policy and masked sequence scoring
- `distillation_trainer.py`
  - Teacher-student KD/CE distillation with strict recipe checks
- `infinite_improver.py`
  - Uncertainty-guided self-improvement loop with quality-gated iteration rollback
- `eval_suite.py`
  - Multi-benchmark evaluation, checkpoint-gated report generation, run-mode metadata, and scorecard generation
- `quant_utils.py`
  - Dynamic int8 quantization + optional TorchScript export
- `hardware_profiles.py`
  - Hardware profile load/show helpers

### 15.2 Core package

- `core/ops.py`: QK-Norm and backend-dispatched attention
- `core/checkpoint_io.py`: secure load/save, hash sidecar and manifest integrity
- `core/config_manager.py`: file/env layered config ingestion and hash snapshots
- `core/config_schema.py`: strict Pydantic section models and cross-field constraints
- `core/dependency_checks.py`: installed-version checks against `requirement.txt`
- `core/hardware_optim.py`: profile detection and backend resolution policy
- `core/lr_schedule.py`: cosine warmup LR helpers
- `core/sequence_ops.py`: shared shifted CE and sequence log-prob math
- `core/training_lifecycle.py`: cadence/sync/LR utility helpers
- `core/logging.py`: shared logger setup

### 15.3 Engine package

- `engine/optimizer_factory.py`
  - AMP policy wrapper
  - optimizer parameter grouping
  - fused AdamW builder

### 15.4 Eval and harness packages

- `eval/benchmark_harness.py`: report-card outputs and disabled external-comparison warning artifacts
- `eval_harness/runner.py`: eval wrapper + benchmark exclusion coverage manifest
- `eval_harness/contamination_checks.py`: source-exclusion coverage helper; not content contamination detection
- `eval_harness/manifests.py`: manifest lifecycle helpers
- `eval_harness/statistical_tests.py`: lightweight mean/std/bootstrap CI

### 15.5 Observability, safety, security, serving

- `observability/metrics.py`: JSONL metric sink
- `observability/training_telemetry.py`: step, gradient, hardware and throughput telemetry
- `observability/security_events.py`: structured security event stream
- `observability/run_registry.py`: lockfile-guarded run registry
- `observability/tracing.py`: span timing helper
- `security/validator.py`: path/data validation and gated code execution
- `safety/prompt_attack_checks.py`: prompt-level attack pattern checks
- `safety/policy_filters.py`: output sanitization and policy check
- `serving/server.py`: fail-closed serving boundary integrating safety checks

### 15.6 Tests

- `tests/test_audit_truthfulness.py`: Pass 1 audit truthfulness checks
- `tests/test_cli_truthfulness.py`: CLI status/deps/report truthfulness checks
- `tests/test_config_profiles.py`: model profile and legacy isolation checks
- `tests/test_core_lr_schedule.py`: LR boundary and token-step equivalence checks
- `tests/test_core_sequence_ops.py`: shifted CE/log-prob helper correctness checks
- `tests/test_data_token_truthfulness.py`: token artifact manifest, source-exclusion, and tokenizer audit checks
- `tests/test_deployment_profiles.py`: deployment tier honesty checks
- `tests/test_eval_integrity.py`: eval checkpoint and benchmark-integrity checks
- `tests/test_hard_evidence.py`: hardware, constrained GPU fit, governance, dedup, and short-validation evidence checks
- `tests/test_kv_cache.py`: bounded local KV cache checks
- `tests/test_model_shapes.py`: dense GQA model shape and generation checks
- `tests/test_quantization_truthfulness.py`: quantization report contract checks
- `tests/test_serving_truthfulness.py`: fail-closed serving checks
- `tests/test_training_safety.py`: preflight, checkpoint, tiny train, and real-token validation checks

---

## 16. Dependency Contract (`requirement.txt`)

- `torch>=2.4.0,<2.9.0`
- `numpy>=1.26.4,<3.0.0`
- `datasets>=2.20.0,<5.0.0`
- `tokenizers>=0.20.3,<0.23.0`
- `regex>=2024.11.6,<2027.0`
- `PyYAML>=6.0.2,<7.0.0`
- `pydantic>=2.10.0,<3.0.0`

---

## 17. Current Caveats and Implementation Realities

Behavior currently present in code and important for future changes:

- In `model.GQAttention`, local/sliding KV cache is bounded to `sliding_window` entries for local layers; global layers still keep full KV.
- Deployment tier estimates are lower bounds only and do not validate GGUF/GPTQ/AWQ/vLLM/llama.cpp support.
- In `model.MoE`, `aux_loss` and `load_balance_loss` are currently the same scalar term in the legacy experimental profile.
- In `core.ops.run_attention`, flash attention exceptions are caught and fallback to SDPA without explicit error surfacing.
- `TrainConfig.dtype` defaults to `bfloat16` while T4 profile prefers `float16`; AMP autocast eligibility does not explicitly gate bf16 by GPU capability.
- CPU profile requests `attention_backend: eager`, but runtime resolver normalizes unknown values to `sdpa`.
- `train.py` includes early stopping by default; uninterrupted token-budget runs require explicit policy alignment.

---

## 18. Verification and Coverage Status

### 18.1 Unit tests present

- LR schedule tests (`tests/test_core_lr_schedule.py`)
- Sequence ops tests (`tests/test_core_sequence_ops.py`)

### 18.2 Current coverage shape

- Strong coverage for shared math utility helpers.
- Limited automated coverage for:
  - full model forward/cache/refinement paths
  - MoE routing edge cases
  - distributed pretrain loop behavior
  - downloader + streaming fault recovery

---

## 19. Kaggle Pretraining Upload Packs

### 19.1 Minimum upload (internet enabled)

- Root:
  - `run.py`, `train.py`, `config.py`, `model.py`, `data.py`, `tokenizer.py`, `hardware_profiles.py`
- Required packages/folders:
  - `core/`
  - `engine/`
  - `observability/`
  - `security/`
  - `safety/`
  - `hardware_configs/`
  - `tokenizer_data/` (must contain `encoder.json` and `merges.json`)
- Dependency spec:
  - `requirement.txt`

### 19.2 Recommended upload (internet disabled)

Everything in 19.1 plus:

- `data_cache/tokens/`
- `data_cache/val_tokens/`
- `data_cache/download_manifest.json`

### 19.3 Optional for resume

- Active checkpoint directory used by `train_cfg.checkpoint_dir`
- Any config files passed via `--config`

### 19.4 Not required for pretrain-only runs

- `alignment.py`
- `sft_trainer.py`
- `distillation_trainer.py`
- `infinite_improver.py`
- `eval_suite.py`
- `eval/`
- `eval_harness/`
- `quant_utils.py`
- `serving/`

---

## 20. Maintenance Checklist for Future Map Syncs

- Re-run module inventory and update counts.
- Re-verify model parameter cardinality after any model/config change.
- Reconcile command list with `run.py` parser choices.
- Reconcile config docs with `config.py` dataclasses and `core/config_schema.py` constraints.
- Reconcile runtime caveats with actual implementation (not intended behavior).
- Reconcile artifact paths with current write locations in trainers/evaluators.
- Keep environment variable section synchronized with all `os.environ` consumers.
