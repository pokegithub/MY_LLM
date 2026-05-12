# Retrieval Source Policy v1

## Purpose

Define which source classes may participate in the first retrieval phase and which must remain out.

## Source classes

### Allowed

Retrieval is allowed only for:

- repo-authored local documentation
- repo-generated local manifests and reports
- explicitly allowlisted local project documents with stable provenance

Allowed sources may be cited. Answers may be given only when supported by valid citations.

### Restricted

Restricted sources are not part of the default first retrieval scope. They may be considered later only with explicit per-source approval and a citation path the verifier can check.

Examples:

- external documents with documented provenance but no retrieval approval yet
- external datasets that may be allowed for limited training roles but are not retrieval-approved

### Blocked

Blocked sources cannot be retrieved or cited for the first retrieval phase.

Examples:

- current `blocked_pending_review` sources in `data_governance/source_license_metadata.json`
- unclear external corpora
- downloaded source content with unresolved provenance or policy status

### Benchmark-excluded

Benchmark-excluded sources are prohibited from retrieval and citation in this phase.

Examples:

- configured benchmark exclusions such as `truthful_qa`, `google/boolq`, `allenai/ai2_arc`, `winogrande`, and `openai/gsm8k`
- all hidden-eval materials under `evals/hidden/`

### Unknown

Unknown sources are not retrievable and not citable. The system must abstain or ask for a valid source.

## Current phase notes

- `allowed_by_repo_policy` for training does **not** imply retrieval approval.
- The current tiny training-readiness allowance does not widen the first retrieval scope.
- Any source outside the allowlisted local-doc/manifests path remains restricted, blocked, benchmark-excluded, or unknown until explicitly reviewed for retrieval use.
