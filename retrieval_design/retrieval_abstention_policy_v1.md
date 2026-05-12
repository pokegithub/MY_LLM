# Retrieval Abstention Policy v1

## Purpose

Keep retrieval from becoming a prettier way to guess.

## Answer with citations

Answer with citations only when:

- the question falls inside the first retrieval scope
- at least one allowed source is available
- the cited material supports the answer

## Partial answer with caveats

Give a partial answer only when:

- some claims are supported
- unsupported parts are explicitly marked unknown or unverified
- the supported claims still carry citations

## Abstain

Abstain when:

- no allowed source supports the claim
- only blocked, benchmark-excluded, or unknown sources are available
- retrieval returns no evidence and the question still requires evidence

## Ask for source

Ask for source when:

- the user wants a source-grounded answer but the relevant source is outside the current retrieval scope
- the available local corpus does not contain the needed material

## Ask for tool

Ask for tool when:

- the answer depends on filesystem inspection, code execution, or another deterministic tool
- freshness or local runtime state matters more than static retrieved text

## `blocked_unverified` vs softer fallback

Use `blocked_unverified` only when a retrieval-required execution path cannot proceed honestly because:

- the retrieval substrate is unavailable or misconfigured
- the only candidate sources are policy-prohibited
- the run contract required retrieval evidence but none can be obtained

Use a softer “need source” or abstention response when the system can still respond truthfully without pretending the answer is verified.

## What must not happen

- unsupported confident answers after empty retrieval
- citation-free answers described as source-grounded
- overconfident use of weak or restricted material
- benchmark-sensitive leakage from excluded material
