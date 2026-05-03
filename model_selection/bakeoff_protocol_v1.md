# Bakeoff Protocol v1

Purpose: choose the main base-model path for this repo using private hidden-eval evidence and Linux-first runtime fit, not public benchmark prestige.

This protocol does not select a model by itself. It defines how the selection must be made later.

## Scope

- Exactly three candidate slots:
  1. current custom base
  2. one open dense 7B-class candidate
  3. one open dense 14B-class candidate
- Dense only
- Open-weight only for the 7B and 14B slots
- No MoE candidates
- No giant-model or cloud-only fantasy candidates

## Required inputs

- `evals/hidden/hidden_eval_spec_v1.json`
- `evals/hidden/hidden_eval_seed_set_v1.jsonl`
- `evals/hidden/hidden_eval_holdout_hash_registry_v1.json`
- `data_governance/allowed_corpus_manifest_v1.json`
- `python run.py status`
- `python run.py deployment-info`

## Hardware assumptions

- Primary bakeoff execution path: Linux-first workstation with roughly 48 GB class GPU memory
- RTX 2050 4 GB is not a serious bakeoff execution target
- Serious cloud or multi-GPU is not required for this protocol package

## Runtime assumptions

- A candidate must be realistically runnable on the intended Linux-first workstation path
- Runtime fit, stability, and structured output reliability are first-class decision inputs
- A model that looks strong on paper but is operationally fragile is not a main-path winner

## Decision rules

- Hidden private evals are the primary evidence source
- Final verified coding success matters more than public benchmark reputation
- Repair-loop success matters more than one-shot polish
- Abstention and source-grounded truthfulness matter
- Runtime fit and deployment fit matter
- Governance readiness still constrains later post-training; this protocol does not grant training permission

## Current custom base rule

- If the current custom base lacks a real usable checkpoint, it is not a full contender
- As of protocol v1, the standard repo checkpoint/status surface does not show usable pretrain, SFT, DPO, or improved checkpoints
- Until that changes, the current custom base remains `requires_real_checkpoint`

## What counts as success

- One candidate ranks highest on the weighted hidden-eval and runtime rubric
- The result is explainable from the recorded evidence
- No candidate is selected by popularity or sentiment

## What counts as failure

- Candidate choice depends mainly on public leaderboard status
- Runtime instability is ignored
- The custom base is kept on the main path without a usable checkpoint or without competitive evidence
- A 14B candidate is favored despite marginal quality lift and clearly worse workstation fit

## What does not belong in this bakeoff

- Public benchmark theater
- MoE comparisons
- Giant-context marketing claims
- Backend integration work
- Model downloads or execution in this package
- Post-training with blocked data
