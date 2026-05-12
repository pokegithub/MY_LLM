# Retrieval Benchmark Exclusion Policy v1

## Purpose

Keep benchmark-sensitive and hidden-eval material out of retrieval so retrieval does not leak holdouts or masquerade as general knowledge.

## What benchmark-excluded means here

Benchmark-excluded material:

- must not be indexed for the first retrieval phase
- must not be queried by the first retrieval phase
- must not be cited in retrieved answers
- must not be used as fallback evidence

## Sources covered

- configured benchmark exclusions from the current repo policy surface
- benchmark-adjacent sources flagged for review
- all hidden-eval artifacts under `evals/hidden/`
- any future hidden/private evaluation documents or answer stores

## Hidden-eval interaction

- `evals/hidden/hidden_eval_holdout_hash_registry_v1.json` is a denylist/audit reference, not a retrieval corpus
- hidden seed items, private targets, and holdout metadata must stay outside retrieval indexing
- future implementation must check both source ids and protected path prefixes before indexing

## Required future implementation checks

Before a document is indexed or queried, future retrieval work must verify:

- source id is not benchmark-excluded
- path is not under `evals/hidden/`
- governance metadata does not mark the source blocked, unknown, or excluded
- the document class is inside the first-scope allowlist

## Why this is strict

The first retrieval phase is meant to improve supported answers, not to create silent contamination or hidden-set leakage.
