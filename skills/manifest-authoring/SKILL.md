---
name: manifest-authoring
applies_to: [ingest]
when: "Drafting a tool manifest from real --help output during `oracle tool add`"
---
You write one tool manifest for a read-only search oracle, from the tool's real `--help` output.
Every fact must come from that help text; never use flags you remember from elsewhere. A later
stage runs your examples against the real tool and drops any op whose example fails.

Ops:
- 2 to 6 ops, each answering a distinct kind of question. Read-only only: never expose anything
  that creates, changes, deletes, syncs, sends or runs something. Subcommands listed as refused
  are mutating; never use them.
- Op names are short and stable, lowercase with dots for grouping: `search`, `get`,
  `jira.search`, `app.history`. Other files refer to them by name.
- Prefer machine-readable output: `--json`, `-o json`, `--output json`, `--format json`.

argv is a list, never a shell string:
- `{p}` is the value of param p; `{?p:--flag,p}` emits `--flag <value>` only when p is set;
  `{?p:--flag=,p}` emits `--flag=<value>` (use it for values that may start with `-`, like
  `-1d`); a literal `--` before a positional lets it start with `-`.
- argv[0] is exactly the binary name given; never a path, never another program.

Params: `type` is one of str, int, bool, list[str], json. Mark required ones; give defaults
for limits; `desc` says what a good value looks like in a few words; `enum` for fixed choices.

returns: your best guess (`format: json`, `path` to the list of records, `id_field`,
`summary_fields`, `pointer` like `jira:{key}`). It is corrected from the real output, so do not
agonise.

when / not_when: one sentence each, in the words people use in questions. `not_when` names the
existing tool to use instead where the two could be confused.

tags: 4 to 8 routing words people would use when they need this tool.

examples: one per op, with harmless values likely to return results on this machine (a common
word, a listing with no filter). `expect: {min_results: 0}` is fine when unsure.
