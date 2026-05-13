# Structured Candidate Contract v1

This contract defines the only backend-generated candidate format accepted by the coding agent. It is a formatting and safety contract, not a correctness claim. The verifier remains authoritative.

## Required Output

Backends must return exactly one JSON object, with no markdown fence, prose, comments, or partial JSON:

```json
{
  "candidate_id": "short-stable-id",
  "summary": "one sentence describing the intended edit",
  "edits": [
    {
      "path": "workspace-relative/path.py",
      "new_content": "full replacement file content"
    }
  ]
}
```

Optional field:

```json
{
  "supported_failure_classes": ["assertion_failure"]
}
```

## Required Fields

- `candidate_id`: non-empty string.
- `summary`: non-empty string. It must not claim verified success.
- `edits`: non-empty list.
- `edits[].path`: workspace-relative path only.
- `edits[].new_content`: non-empty full replacement file content.

## Rejected Cases

- plain prose
- markdown-wrapped JSON
- partial JSON
- missing required fields
- empty `candidate_id`, `summary`, `edits`, or `new_content`
- absolute paths
- parent traversal such as `../file.py`
- output that cannot validate into the existing `Candidate` and `FileEdit` types

## Malformed-Output Retry Policy

The local Transformers backend may make at most two format-repair attempts after malformed or schema-invalid output. Each retry prompt must include the failure class, parse/schema message, the same original task context, and this contract. Retries are formatting attempts only; they do not count as success unless the final output validates through the same parser.

If all attempts fail, the backend surfaces `malformed_candidate_output` or `schema_validation_failed`. The agent must not apply edits and must not report verified success.

## Truth Limits

- Schema-valid output is not correct code.
- Tiny-model formatting success is not coding ability.
- Backend smoke is not bakeoff evidence.
- `quality_claim` remains `none` unless verification passes.
