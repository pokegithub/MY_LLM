# Benchmark Holdout Policy v1

Purpose: keep benchmark-risk and hidden-eval material out of future training inputs unless a later policy change is explicitly documented. This is a repo-side training-readiness artifact, not a legal opinion.

## 1. What counts as benchmark-risk

A source is benchmark-risk or benchmark-adjacent if any of the following are true:

- It is explicitly listed in `train_cfg.excluded_benchmark_sources`.
- It is a known benchmark or benchmark split.
- It is a benchmark-like derivative, contamination probe, or evaluation-oriented corpus.
- Its upstream mixture/provenance suggests likely overlap with benchmark tasks, but the repo cannot yet verify that overlap cleanly.

## 2. Current explicit benchmark-source exclusions

The current repo configuration excludes these sources from training:

- `truthful_qa`
- `google/boolq`
- `allenai/ai2_arc`
- `winogrande`
- `openai/gsm8k`

These are source-risk exclusions only. They are not content-level contamination detection.

## 3. Hidden eval exclusion rule

All hidden eval assets under `evals/hidden/` are excluded from:

- training
- training exports
- preference generation
- retrieval indexes used for model answering

Relevant private artifacts currently include:

- `evals/hidden/hidden_eval_seed_set_v1.jsonl`
- `evals/hidden/hidden_eval_holdout_hash_registry_v1.json`
- `evals/hidden/private/hidden_eval_private_targets_v1.json`

## 4. Interaction between holdout registry and training-source policy

Before a source enters the tiny allowed subset, the repo should check:

1. documented license evidence exists
2. the source is not in configured benchmark exclusions
3. the source is not itself a hidden-eval asset
4. the source does not directly reuse private hidden-eval prompts, targets, or fixtures
5. intended training role is documented
6. `legal_clearance_claim` remains `none`

The hidden-eval holdout hash registry is an audit tool for private eval assets. It does not prove that broader training corpora are contamination-free.

## 5. What this policy does not claim

- It does not provide legal clearance.
- It does not prove content-level benchmark decontamination.
- It does not prove that a documented license string is sufficient for training use.
- It does not make a blocked source safe to train.

## 6. Repo-policy rule for Phase A readiness

If a source is benchmark-excluded, hidden-eval-linked, or still unclear enough that contamination or provenance cannot be described honestly, keep it blocked or excluded.
