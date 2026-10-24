---
name: jql
applies_to: [planner, check]
scope: tool:acli
when: "Writing JQL for acli jira.search"
---
JQL for acli `jira.search`:
- Fields: `project`, `status`, `assignee`, `reporter`, `priority`, `issuetype`, `labels`,
  `updated`, `created`, `resolution`, `text` (full text), `summary`.
- People: `assignee = currentUser()`; a display name in quotes: `assignee = "Andrew Neville"`.
- Text: `text ~ "ownership transfer"` (all words, any order). Quote multi-word values.
- Time: `updated >= startOfDay()`, `updated >= -1d`, `created >= startOfWeek()`, `-2w`.
- Open work: `resolution = Unresolved` or `status != Done`; lists: `status in ("To Do", "In Progress")`.
- Combine with AND / OR and parentheses; end with `ORDER BY updated DESC`.
- Avoid history operators (`was`, `changed`) unless the question is about history.
