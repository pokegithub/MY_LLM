# Candidate Shortlist v1

Purpose: define the first pre-bakeoff candidate slots without selecting a winner or claiming model quality.

This shortlist is not a bakeoff result. It only defines which model paths must become locally available and pass readiness gates before full hidden-eval execution.

## Candidate Groups

1. `repo_custom_base_candidate`
   - Role: current custom base candidate.
   - Status: `requires_real_checkpoint`.
   - Bakeoff eligibility: blocked until a real usable checkpoint exists.
   - Quality claim: none.

2. `open_dense_7b_candidate_slot`
   - Role: one dense open-weight 7B-class candidate slot.
   - Status: `proposed_not_selected`.
   - Local path placeholder: `./run_artifacts/local_models/candidates/open-dense-7b`.
   - Bakeoff eligibility: blocked until a reviewed local checkpoint passes readiness smoke.
   - Quality claim: none.

3. `open_dense_14b_candidate_slot`
   - Role: one dense open-weight 14B-class candidate slot.
   - Status: `proposed_not_selected`.
   - Local path placeholder: `./run_artifacts/local_models/candidates/open-dense-14b`.
   - Bakeoff eligibility: blocked until a reviewed local checkpoint passes readiness smoke.
   - Quality claim: none.

## Rules

- Dense only.
- No MoE candidates in Phase A readiness.
- No winner is selected by this artifact.
- Local availability is not quality.
- Structured JSON compliance is not code correctness.
- Full bakeoff remains unexecuted.
- The current custom base cannot masquerade as a full contender without a real usable checkpoint.
