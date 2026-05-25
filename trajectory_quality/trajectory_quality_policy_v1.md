# Trajectory Quality Policy v1

## Purpose

Stored trajectories are useful only when their evidence matches their future use. This policy prevents exact-tool, fixture, scripted, backendless, blocked, or thin-provenance traces from being treated as positive model-training demonstrations.

This policy does not train a model, improve weights, or claim learning.

## Future-use buckets

### SFT-positive eligibility

Use only when all of the following are true:

- `operation=solve`
- `status=verified_success`
- `quality_claim=verification_passed`
- meaningful verifier evidence exists
- task shape is not a fixture or test-harness artifact
- coding tasks used a real model backend, not scripted/stub/backendless generation
- touched files and provenance are sufficient to explain the final behavior

Exact deterministic tasks are not general coding SFT examples.

### Preference-winner eligibility

Use only when:

- final status is `verified_success`
- a kept winner attempt passed verification
- at least one loser attempt from the same trajectory failed verification
- ranking reason is verifier-backed, not style-based
- the trajectory is not fixture-only or scripted-only

### Preference-loser eligibility

Use only when:

- a failed or regressing attempt has meaningful verifier failure evidence
- the loser can later be paired with a verified winner from the same task lineage
- the run is not being presented as successful

Blocked runs are never preference winners.

### Retrieval-memory eligibility

Allowed when the record is useful as a labeled memory item:

- verified solutions
- exact-tool examples
- failed attempts with failure class evidence
- blocked states with a clear blocked reason

Retrieval-memory eligibility must preserve outcome labels and must not imply model improvement or legal approval.

### Failure-analysis eligibility

Allowed for failed, blocked, or regressing traces when:

- the failure reason is explicit
- verifier or blocked-state evidence exists
- the record is not being turned into a positive demonstration

### Exact-tool curriculum eligibility

Allowed for deterministic exact-tool traces only. These may teach routing or exactness, but they must not be mixed into general coding-reasoning SFT.

## Risky or rejected classes

### Fixture-only

Test-harness or fixture traces are useful for testing orchestration. They are not positive training data.

Backend smoke fixtures, candidate-readiness smoke workspaces, and tiny verifier
targets such as `backend_smoke_target.py` are fixture-only even when a real
backend reaches a verifier pass. They prove integration reachability, not
general coding quality.

Private hidden-eval seed workspaces under `run_artifacts/hidden_eval_runs/`
are also fixture/holdout material. They may provide evaluation evidence, but
they must not become SFT-positive training demonstrations.

### Backendless blocked

No-backend coding traces are useful as blocked-state evidence only. They are not model-solution demonstrations.

### Insufficient provenance

Records missing run id, operation, route, status, task text, or evidence fields cannot be used for training-oriented exports.

### Rejected low value

Planning-only, scripted-only, unsupported, or otherwise thin records may remain in the store for auditability, but should not drive model training.

## Current-store warning

If the current trajectory store is mostly exact-tool, fixture, scripted, or backendless traces, the audit must say so directly. Counts must not be inflated for appearance.
