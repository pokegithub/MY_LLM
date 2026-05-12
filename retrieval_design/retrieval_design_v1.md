# Retrieval Design v1

## Purpose

Define the first retrieval scope for this repo so future retrieval work reduces unsupported answers instead of making them sound better.

## Status

- Planned only
- Not implemented
- Not a retrieval capability claim

## First retrieval scope

The first retrieval scope is intentionally narrow:

- repo-authored local documentation
- repo-generated local manifests and reports
- explicitly allowlisted local project documents with stable filesystem provenance

The first scope does **not** include:

- downloaded training corpora
- benchmark datasets
- hidden eval artifacts
- broad web content
- memory stores
- vector or semantic retrieval

## Why the first scope is narrow

The current repo has a tiny training-readiness allowance and strict benchmark exclusions. It does **not** yet have enough provenance, source-policy coverage, or verifier support to justify broad document retrieval. Starting with repo-authored and repo-generated local materials keeps citations checkable and keeps benchmark-sensitive or unknown material out of the first path.

## What retrieval is supposed to improve

- source-grounded answers about repo behavior and local project state
- explicit citation discipline
- abstention when no supported source exists
- future hidden-eval performance on retrieval-grounded QA

## What retrieval must not be allowed to do

- act as a magic fallback for unsupported answers
- convert uncited answers into source-grounded answers
- bypass verifier standards
- index benchmark-excluded or hidden-eval material
- imply freshness, legal approval, or verification that the retrieved text does not actually support

## Primary target environment

- Linux-first workstation path for future integration
- repo-local filesystem documents only for the first retrieval scope

## Explicit deferrals

- retrieval implementation
- embeddings or vector search
- web search
- memory systems
- benchmark ingestion
- cross-project document federation
- automatic semantic ranking
