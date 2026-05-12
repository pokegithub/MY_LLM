# Retrieval Verifier Contract v1

## Purpose

Define what the verifier must eventually check when retrieval-backed answers are introduced.

## Verifier authority

The verifier remains authoritative. Retrieval does not create a new success path outside verifier checks.

## Required verifier checks

- citation presence when the answer claims to be source-grounded
- source-class policy compliance
- locator resolvability
- claim-support alignment between answer text and cited evidence
- contradiction handling across cited sources
- explicit handling of unsupported claims

## Verification outcomes

### Verification pass

Allowed only when:

- every asserted source-grounded claim has valid citations
- cited sources are policy-allowed for retrieval
- the cited material supports the claims
- no hidden benchmark or holdout material was used

### Verification failure

Use verification failure when:

- a source-grounded answer lacks required citations
- citations are malformed or insufficient
- the answer overclaims beyond the cited evidence
- cited sources contradict the answer materially

### Blocked or unverified

Use blocked or unverified when:

- retrieval was required by the run contract but the retrieval path was unavailable
- only prohibited sources were available
- the requested answer required source-grounded evidence that the system could not access honestly

## Deferred items

- actual retriever execution checks
- semantic-ranking diagnostics
- freshness-aware web retrieval checks
- production latency thresholds
