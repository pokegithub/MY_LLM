# Backend Failure Contract v1

Purpose: define how backend-related failures must surface without weakening the current exact-tool, verifier, or fail-closed semantics.

## Failure classes

### `backend_not_configured`
- Report as: `blocked_unverified`
- Retry in same solve: no
- Truthful report still possible: yes
- Must never be misreported as success: yes

### `backend_unavailable`
- Report as: `blocked_unverified`
- Retry in same solve: no for first integration path
- Truthful report still possible: yes
- Must never be misreported as success: yes

### `model_load_failed`
- Report as: `blocked_unverified`
- Retry in same solve: no
- Truthful report still possible: yes
- Must never be misreported as success: yes

### `malformed_candidate_output`
- Report as: `blocked_unverified` after bounded malformed-output retry exhaustion for initial-candidate failure, or failed attempt with explicit critique if it happens mid-loop
- Retry in same solve: only through the explicit bounded malformed/schema formatting retry policy; retries are logged formatting attempts, not progress
- Truthful report still possible: yes
- Must never be misreported as progress: yes

### `schema_validation_failed`
- Report as: `blocked_unverified`
- Retry in same solve: only through the explicit bounded malformed/schema formatting retry policy; after retry exhaustion fail closed
- Truthful report still possible: yes
- Must never be misreported as success: yes

### `runtime_execution_failed`
- Report as: `blocked_unverified`
- Retry in same solve: no by default
- Truthful report still possible: yes
- Must never be misreported as success: yes

### `timeout`
- Report as: `blocked_unverified`
- Retry in same solve: no by default
- Truthful report still possible: yes
- Must never be misreported as success: yes

### `unsupported_capability`
- Report as: `blocked_unverified`
- Retry in same solve: no
- Truthful report still possible: yes
- Must never be misreported as success: yes

### `degraded_mode_only`
- Report as: `blocked_unverified` if the missing capability prevents machine-readable candidate generation
- Retry in same solve: no
- Truthful report still possible: yes
- Must never be misreported as a usable backend integration: yes

### `verification_failed_after_candidate`
- Report as: `verification_failed`
- Retry in same solve: yes, if the bounded repair loop still has budget and the backend can provide a repair candidate
- Truthful report still possible: yes
- Must never be misreported as success: yes

## Invariants

- backend health must not affect deterministic exact-task routing
- verifier remains the final authority on success
- bounded malformed/schema retries do not count as progress or success
- malformed or unavailable backend output is not partial success
- a truthful report must still be written whenever possible
