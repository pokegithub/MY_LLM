# Backend Capability Gate v1

This gate defines the minimum bar for a real backend before wiring it into the agent loop is worth doing.

## Mandatory gates

1. **Structured candidate output validity**
   - Check: backend can produce one bounded candidate with required fields
   - Pass: output maps cleanly to `Candidate` and `FileEdit` without manual repair
   - Fail: prose-only output, missing fields, or invalid edit structure

2. **Schema adherence / parse reliability**
   - Check: repeated seed-task generations remain parseable
   - Pass: at least 95% parseable on the seed gate set
   - Fail: malformed output is frequent enough to dominate the agent loop

3. **Coding usefulness beyond the no-backend baseline**
   - Check: backend does better than the current blocked/no-backend path on hidden coding seed tasks
   - Pass: at least one non-fixture hidden coding task reaches `verified_success`, and blocked/no-backend behavior is materially improved
   - Fail: backend produces only blocked, malformed, or verifier-red outcomes

4. **Clean failure surfacing**
   - Check: load/runtime/output failures map to explicit backend failure classes
   - Pass: failure is reported clearly and never misreported as solve success
   - Fail: backend errors are swallowed, blurred, or converted into fake progress

5. **Deterministic exact-task bypass preserved**
   - Check: exact symbolic tasks still skip backend generation entirely
   - Pass: backend health does not affect deterministic exact-task handling
   - Fail: exact tasks start depending on backend availability

6. **Verifier compatibility**
   - Check: backend candidate outcomes still require verifier evidence before success
   - Pass: no backend candidate can become `verified_success` without a green `VerificationReport`
   - Fail: model output is treated as enough by itself

7. **Retry-loop compatibility**
   - Check: backend can supply an initial candidate and either a narrower repair candidate or an explicit refusal
   - Pass: repair attempts remain machine-readable and bounded
   - Fail: repair path is ambiguous, prose-only, or incompatible with current attempt tracking

8. **Runtime stability on Linux-first workstation path**
   - Check: loading and generation are stable enough on the intended workstation target
   - Pass: no repeated model-load crashes, persistent OOMs, or unusable instability during seed evaluation
   - Fail: runtime instability dominates the loop

## Optional first-integration gate

9. **Optimization compatibility**
   - Check: backend can optionally supply an optimization candidate after a verified green result
   - Pass: optimizer path works, or backend explicitly declines optimization cleanly
   - Fail: optimizer path is falsely claimed or breaks verified-good rollback behavior

The first integration may proceed without optimization support, but not without explicit clean decline behavior.
