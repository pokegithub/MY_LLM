# Celadon Runtime Matrix

## Retrieval commands
- retrieval-answer exists for local repo-doc questions.
- retrieval-search exists for lexical search over approved local docs.
- memory-sync command does not exist in Phase A.

## Source policy
- Missing evidence must produce abstention instead of guessing.
- Hidden private targets are excluded from retrieval and must not be cited.
- Source-grounded answers require citations.

## Telemetry caveat
- The telemetry bridge may emit delayed metrics during archive replay.
- The telemetry bridge is not guaranteed to emit metrics instantly.

## Partial restore evidence
- Audit-log export is available for status summaries.
- Per-tenant restore timing evidence is not present in this note.
