# Backend Integration Plan v1

Purpose: define the first honest model backend path for the verified coding agent without weakening exact-tool routing, verifier authority, or fail-closed behavior.

## What backend integration is supposed to unlock

- real model-generated coding candidates inside the existing agent loop
- real hidden-eval and bakeoff execution later
- meaningful repair-loop and optimizer testing with a non-scripted backend

## What it must not break

- deterministic exact-task bypass
- verifier authority over final success
- blocked/unverified truthfulness
- machine-readable candidate boundaries

## Chosen first backend path

The first integration target is:

- **one Linux-first in-process local Transformers backend**
- one model loaded directly inside the Python process
- one structured `Candidate` contract from `agent/types.py`

This path is intentionally narrower and less flashy than a service-based path. It is chosen because it is the safest first integration:

- no extra server/process boundary is required
- no OpenAI-compatible shim is required
- fewer moving parts means clearer failure reporting
- structured output validation can happen at the point of candidate creation
- it fits the repo’s fail-closed style better than a more elaborate serving path

## Why only one path for now

- multiple backend kinds would broaden scope before a single truthful path exists
- a first integration should minimize fake progress from infrastructure complexity
- the bakeoff package already says runtime fit matters; one narrow path is enough to test that honestly

## Why alternatives are deferred

- **vLLM or service-style serving**: deferred until a local in-process path proves structured candidate reliability and verifier compatibility
- **remote API or cloud-only backend**: deferred because it adds network/deployment variables before the local control loop is validated
- **Windows-first backend work**: deferred because the Phase A target path is Linux-first
- **multi-backend abstraction**: deferred because `agent/backend.py` already provides a narrow protocol and does not need expansion yet

## Exact integration scope

- add one future concrete backend implementation under `agent/backend.py`
- keep `CodingModelBackend` as the core contract
- keep `agent/orchestrator.py` thin: load backend, request candidates, verify, report
- keep `run.py` thin: no broad CLI expansion

## Explicit deferrals

- no backend implementation in this package
- no model download
- no serving redesign
- no runtime matrix
- no multi-backend support
- no production deployment work
