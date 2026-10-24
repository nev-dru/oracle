---
name: cql
applies_to: [planner, check]
scope: tool:acli
when: "Writing CQL for acli confluence.search"
---
CQL for acli `confluence.search`:
- `text ~ "salesforce cloud"` searches title and body; `title ~ "runbook"` titles only.
- `space = ARCH` narrows to a space; `type = page`.
- Combine with AND; keep the words to the subject ("ingestion gateway"), not the question.
- Design docs, runbooks and decisions live here; tickets do not.
