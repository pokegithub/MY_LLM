# Output Normalization Policy v1

This policy defines the only formatting normalization allowed before backend
candidate schema validation. It exists to handle one common model formatting
mistake without weakening `structured_candidate_contract_v1`.

Normalization is not silent. Every attempt must be reported with parse and
normalization fields, and every normalized payload must still pass the exact
same `Candidate` and `FileEdit` validation path. Verifier authority is
unchanged.

## Allowed Normalization

The backend parser may strip one outer markdown code fence only when all of the
following are true:

- the raw output contains exactly one outer fenced block
- there is no prose before or after the fence except whitespace
- the opening fence has no language tag or uses `json`
- the fenced body is pure JSON
- the parsed payload validates against `structured_candidate_contract_v1`
- schema validation, safe-path validation, and non-empty edit validation still
  run after normalization

When this case succeeds, reports must include:

```text
raw_parse_status: failed
normalization_attempted: true
normalization_applied: true
normalization_kind: markdown_fence_stripped
final_parse_status: passed
schema_validation_status: passed
```

## Rejected Normalization Cases

The parser must reject:

- prose before or after fenced JSON
- multiple fenced blocks
- nested or inner markdown fences
- partial JSON
- fenced non-JSON
- markdown lists containing JSON
- JSON mixed with explanation
- missing required candidate fields
- empty edit lists
- empty `new_content`
- unsafe paths

Rejected normalization must be reported through
`normalization_rejected_reason`. The agent must not apply edits when
normalization is rejected.

## Path Handling

Paths are stricter than markdown fences.

- Workspace-relative paths are preferred and required.
- Absolute paths are rejected by default.
- Parent traversal is rejected.
- Home, temp, system, or drive-root paths outside the workspace are rejected.
- This policy does not implement absolute-to-relative conversion.

Future absolute-to-relative conversion may only be considered if all of these
are true: the path resolves inside the active workspace, the target is an
allowed task file, conversion is logged explicitly, and the converted relative
path still passes safe-path validation.

## Examples

Accepted after logged normalization:

````text
```json
{"candidate_id":"fix","summary":"repair value","edits":[{"path":"demo.py","new_content":"def value():\n    return 2\n"}]}
```
````

Rejected because prose surrounds the fence:

````text
Here is the JSON:
```json
{"candidate_id":"fix","summary":"repair value","edits":[{"path":"demo.py","new_content":"def value():\n    return 2\n"}]}
```
````

Rejected because the path is absolute even though the fence can be stripped:

````text
```json
{"candidate_id":"fix","summary":"repair value","edits":[{"path":"C:/repo/demo.py","new_content":"def value():\n    return 2\n"}]}
```
````

## Truth Limits

- Normalized JSON was not originally compliant output.
- Schema-valid output is not correct code.
- A verifier pass on a tiny fixture is not general coding ability.
- Backend smoke is not bakeoff evidence.
- `quality_claim` remains `none` unless verifier evidence says otherwise.
