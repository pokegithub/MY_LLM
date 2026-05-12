# Citation Contract v1

## Purpose

Define the first checkable citation format for retrieved answers in this repo.

## Citation rule

Any answer presented as source-grounded must include one or more checkable citations. An uncited answer is not a source-grounded answer.

## Citation granularity

- claim-level support
- one citation may support one claim or a tightly related group of claims
- multi-source support is allowed when synthesis is required

## Required citation fields

- `citation_id`
- `source_class`
- `source_id`
- `document_ref`
- `locator_type`
- `locator`
- `support_kind`
- `support_snippet_hash`

## Allowed locator types for the first phase

- `line_range`
- `section_heading`
- `json_pointer`

## What counts as insufficient citation

- missing any required field
- citing a blocked, benchmark-excluded, or unknown source
- citing a document without a resolvable locator
- citing text that does not actually support the claim
- vague phrases like “according to sources” without a concrete citation object

## Important distinction

- cited answer: has required citation structure
- uncited answer: has no valid citation structure and cannot count as source-grounded
- cited but insufficiently supported answer: has citations, but the citations do not support the claim strongly enough

## Limitation

A citation is an evidence reference, not proof by itself. The verifier still has to check source policy, citation structure, and support strength.
