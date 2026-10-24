# Recipes to rebuild on the work laptop

Recipes live beside the registry they call: `registry/recipes/*.yaml` for `registry/tools/`.
This folder starts empty because a recipe names ops, and the work tools get their op names
from `oracle tool add`. The four seeded recipes were written for the home tools in
`profiles/home/recipes/`; rebuild each one here once the tools it calls are active.

## Order of work

1. Add the tools with `oracle tool add` and `oracle tool activate` (see `docs/DESIGN.md`,
   "Tool ingestion workflow").
2. Copy a recipe from `profiles/home/recipes/` into this folder.
3. Rename each stage's `op` to the real ingested op, and fix its params to match that op.
4. Run `oracle tool list`. A stage whose tool is missing shows as a note
   ("runs without ... not active here"); a wrong op or param is a problem. Fix every problem.
5. Run `oracle eval`. All four golden cases use these recipes.

Golden cases name ops directly (`golden.yaml`). Keep these names when ingesting, or update
`golden.yaml` on purpose: `code-search.search`, `code-graph.find_symbol`, `code-graph.expand`,
`code-graph.path_between`, `acli.jira.search`, `acli.confluence.search`,
`workiq.people.lookup`, `workiq.teams.search`, `workiq.calendar.today`,
`argocd.app.history`, `kubectl.events`.

## The four chains

**`code.find_and_expand`** ("where does X happen in code"): `code-search.search` →
`code-graph.symbols_at` → `code-graph.expand(callers)`; falls back to
`code-graph.find_symbol` when the search finds nothing.
- `symbols_at` is a home-only bridge: vex returns chunk ids, not graph symbol ids. Check
  what `csearch` returns. If its results carry graph symbol ids, drop the `symbols` stage and
  bind `expand` to `$search.ids`. If not, the work graph DB needs an equivalent
  path:line → symbol op.

**`code.trace_path`** ("how A talks to B"): `code-graph.find_symbol(A)` ∥
`code-graph.find_symbol(B)` → `code-graph.path_between`, plus `code-search.search` for both
names and `acli.confluence.search` for a design doc.
- `path_between` binds to `results[0].name`. Check which field the work graph's
  `path_between` expects (a name or a qualified id) and bind that one.
- At home "cloud" matched no symbol, so the completeness check had to replan. Expect the same
  at work when a name is a concept rather than a symbol.

**`people.profile`** ("who is X"): `workiq.people.lookup` ∥
`acli.jira.search(assignee)` ∥ `workiq.teams.search(sender, last 14 days)`; breadth.
- Real Jira often wants an account id or email for `assignee`, not a display name. If
  `assignee = "Andrew Neville"` returns nothing, chain it after the lookup:
  `assignee = "$directory.results[0].email"` (or whichever field the work directory returns).
- Check that `workiq.teams.search` really filters by sender and takes `-14d`.

**`today.events`** ("important events from today"): `workiq.calendar.today` ∥
`acli.jira.search(updated >= startOfDay())` ∥ deploys ∥ cluster warnings ∥
`workiq.teams.search(since today)`; breadth.
- Home uses `argocd.app.list` for deploys because the fake's `app.history` needs an app name,
  and Warning-level `kubectl.events` because the fake has no time filter. So neither is
  limited to today. At work, prefer a deploy op that takes a time (`argocd.app.history`
  with `since: "{window_start}"`, if the real CLI supports it) and `kubectl.events` with a
  `since` param, if the ingested op has one.
- `{window_start}` and `{window_end}` are the resolved "today" bounds (ISO timestamps).

## Rules that apply to every recipe

- A stage's params use the recipe's own params (`{name}`), the time window (`{window_start}`),
  or earlier stages (`$search.ids`, `$directory.results[0].email`). The loader rejects a
  typo in either.
- Bindings and templates are checked when recipes load, so `oracle tool list` is the fast
  feedback loop. Every change still has to pass `oracle eval`.
- Write `when` lines for the shape of the question, not the golden wording.
