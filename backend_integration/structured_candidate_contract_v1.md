# Structured Candidate Contract v1

This contract defines the only backend-generated candidate format accepted by the coding agent. It is a formatting and safety contract, not a correctness claim. The verifier remains authoritative.

## Required Output

Backends must return exactly one JSON object, with no markdown fence, prose, comments, or partial JSON. `output_normalization_policy_v1` may strip one outer markdown JSON fence before validation, but that is logged as normalization and is not considered originally compliant output:

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
- markdown-wrapped JSON outside the narrow `output_normalization_policy_v1`
- partial JSON
- missing required fields
- empty `candidate_id`, `summary`, `edits`, or `new_content`
- absolute paths
- parent traversal such as `../file.py`
- output that cannot validate into the existing `Candidate` and `FileEdit` types

## Output Normalization

`output_normalization_policy_v1` allows only one normalization case: exactly one
outer markdown fence with optional `json` language tag and no prose outside the
fence. The fenced body must be pure JSON and must pass this same contract after
the fence is stripped. Prose, multiple fences, partial JSON, unsafe paths, and
empty edits remain rejected.

Absolute path conversion is not implemented. Absolute paths remain rejected by
default even when markdown-fence normalization succeeds.

## Malformed-Output Retry Policy

The local Transformers backend may make at most two format-repair attempts after malformed or schema-invalid output. Each retry prompt must include the failure class, parse/schema message, the same original task context, and this contract. Retries are formatting attempts only; they do not count as success unless the final output validates through the same parser.

If all attempts fail, the backend surfaces `malformed_candidate_output` or `schema_validation_failed`. The agent must not apply edits and must not report verified success.

## Truth Limits

- Schema-valid output is not correct code.
- Tiny-model formatting success is not coding ability.
- Backend smoke is not bakeoff evidence.
- `quality_claim` remains `none` unless verification passes.
