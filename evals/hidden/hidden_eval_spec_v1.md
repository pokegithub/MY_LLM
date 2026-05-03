# Hidden Eval Spec v1

This artifact defines the first private hidden evaluation backbone for Phase A.

It is a decision tool, not a public benchmark. It exists to compare:
- base-model candidates
- backend integrations
- retrieval/citation behavior
- verifier-loop quality

It does not claim model improvement by itself.

## Operating rules

- Keep answer keys and hidden behavior targets out of casual help text.
- Exclude all hidden items from training, export, and retrieval corpora.
- Keep one frozen anchor split for comparability and one rotating split for refresh.
- Prefer exact or structured scoring whenever possible.
- Treat unsupported confidence as failure when the task requires a tool, source, or abstention.

## Categories

### 1. Exact symbolic correctness
- Purpose: check exact output, exact file lookup, and deterministic tool-routing behavior.
- Success: exact output matches the private target or the system routes to the exact tool path correctly.
- Failure: wrong exact output, unsupported freeform guessing on exact tasks, or missing required exact evidence.
- Scoring: exact only.
- Evidence required: exact output or deterministic tool result.
- Hidden material: expected outputs and file-position targets.
- Contamination handling: keep all prompts and targets out of training/export corpora; rotate only through the registry.
- Not measured yet: long-horizon reasoning style, chain-of-thought quality, or public benchmark rank.

### 2. Abstention / need-tool / need-source behavior
- Purpose: measure whether the system asks for a source, asks for a tool, or abstains when evidence is insufficient.
- Success: correct abstain / need-tool / need-source decision with no unsupported confident answer.
- Failure: guessing, unsupported certainty, or answering freshness-sensitive questions from stale evidence.
- Scoring: structured label first; answer text is secondary.
- Evidence required: explicit abstention or routing/request decision.
- Hidden material: expected decision label and freshness trigger.
- Contamination handling: stale-document and tool-required cases remain excluded from training corpora.
- Not measured yet: politeness style, verbosity, or user-preference phrasing.

### 3. Source-grounded truthfulness
- Purpose: test whether answerable questions are answered from provided evidence with explicit grounding.
- Success: answer matches the document-supported claim set and cites the required source.
- Failure: unsupported claim, missing citation, or contradiction of the provided source.
- Scoring: structured claim check plus citation presence.
- Evidence required: answer with citation bundle.
- Hidden material: accepted claim fragments and citation expectations.
- Contamination handling: keep source docs and accepted-claim targets in the hidden set registry.
- Not measured yet: stylistic elegance or broad world knowledge outside the supplied source.

### 4. Coding patch success
- Purpose: measure one-pass patch correctness on a small hidden behavior target.
- Success: minimal patch satisfies all hidden behavior assertions.
- Failure: verifier failure, incorrect behavior, or broad unrelated rewrite.
- Scoring: hidden behavior assertions only.
- Evidence required: patched file plus verifier report or hidden behavior report.
- Hidden material: behavior assertions and expected edge-case outputs.
- Contamination handling: hidden fixtures and assertions remain excluded from training/export corpora.
- Not measured yet: large-scale refactors, benchmark pass@k, or subjective code style.

### 5. Coding repair-loop success
- Purpose: measure whether the solve-plus-repair loop reaches a verified fix within budget.
- Success: final repair passes all hidden assertions within the retry budget.
- Failure: retry exhaustion, regression, or unverifiable patch.
- Scoring: hidden assertions plus bounded repair outcome.
- Evidence required: attempt history, final verifier result, and hidden behavior report.
- Hidden material: edge-case assertions and retry-sensitive behavior checks.
- Contamination handling: hidden repair fixtures stay private and rotate through the registry.
- Not measured yet: multi-candidate search quality or RL-style exploration metrics.

### 6. Retrieval-grounded QA
- Purpose: measure document retrieval plus citation-backed answering on multi-document questions.
- Success: answer is supported by the retrieved docs and includes the required citation bundle.
- Failure: uncited answer, unsupported synthesis, or failure to acknowledge missing evidence.
- Scoring: structured answer support and citation validation.
- Evidence required: answer text, retrieved document references, and citations.
- Hidden material: accepted synthesis targets and required supporting documents.
- Contamination handling: retrieval docs are excluded from training/export corpora and tracked by hash.
- Not measured yet: semantic search quality beyond the hidden seed, web search quality, or recall-at-scale marketing metrics.

### 7. Verifier-loop KPIs and basic latency/cost tracking
- Purpose: measure whether solve reports contain enough operational evidence to compare candidate systems honestly.
- Success: solve report contains attempt count, verification summary, elapsed timing, and per-check evidence fields required by the target.
- Failure: missing operational fields, unverifiable cost/latency accounting, or green status without verifier evidence.
- Scoring: structured report-field presence and consistency.
- Evidence required: solve report with attempt, verification, and timing fields.
- Hidden material: required field set and bounded expectations for the target task.
- Contamination handling: KPI audit items remain excluded from training/export corpora.
- Not measured yet: cluster-scale throughput, production traffic mix, or long-run cost optimization.

## Frozen vs rotating policy

- Frozen split:
  - anchor items for longitudinal comparison
  - change only with a version bump or retirement record
- Rotating split:
  - refreshable items for contamination resistance
  - rotate through the holdout registry with explicit active/retired state

## What this package does not do

- It does not execute a bakeoff.
- It does not implement retrieval.
- It does not train a model.
- It does not create a public score table.
- It does not prove contamination resistance beyond registry discipline and explicit exclusion rules.
