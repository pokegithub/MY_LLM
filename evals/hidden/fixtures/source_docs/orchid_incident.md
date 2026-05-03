# Orchid Incident Note

Date: 2026-04-15
Service: Orchid API
Window: 01:30-02:15 UTC
Severity: SEV-2

Primary cause:
- cache invalidation worker was disabled during a config rollout

Customer impact:
- stale account balances were visible in dashboard reads
- write path requests were unaffected

Mitigation:
- re-enabled the invalidation worker
- replayed the invalidation backlog
