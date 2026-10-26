# Finishing the oracle on the work laptop

Paste this into Claude Code at the repo root on the work laptop:

> Read CLAUDE.md and docs/DESIGN.md, then docs/WORK-LAPTOP.md, and work through WORK-LAPTOP.md one phase at a time, starting with phase 1. At the end of each phase, report what passed and what is next, and wait for me before the next phase.

## Where things stand (2026-10-26)

- M1 to M4 are built and `make gate-m4` is green at home: 317 tests, 26/26 manifest examples, ranker top-3 97%, golden 7/7 (with the two name follow-up cases), about 29K tokens and 27 s per case, about 200K tokens in all.
- Done at home before the move: golden checks per profile (`profiles.<name>`, and `make accept` for work); manifests that record the tool's version for drift; the registry, recipes and two eval files read from `~/.oracle/`, so internal data never sits in the repo.
- At home every work tool is a fake: acli and workiq are fake CLIs (`tests/fakes/bin/`), kubectl, argocd, es and mongo were added through `oracle tool add` against fakes, and code search runs over the acme fixture repo. The argv shapes in `profiles/home/tools/` are guesses.

## Nothing goes back

Manifests, recipes, real pointers and work queries are internal, and this clone never pushes (DESIGN.md, "On the work laptop, nothing goes back"). Code goes one way: the personal laptop pushes, this laptop pulls.

- Internal data lives under `~/.oracle/`, never in tracked files: the registry (`registry/tools/`) and its recipes (`registry/recipes/`), `golden.yaml` (real pointers as `profiles.default` blocks, and cases of its own), `ranker_eval.yaml` (work ranker cases) and `notes/`. For history, `~/.oracle` can be a local git repo with no remote.
- `main` stays exactly as pulled: `git status` is clean, and updates come in with `git pull --ff-only`.
- A code fix that real tools force goes on a local branch, `work-fixes`, so work is not blocked. Write it up in `~/.oracle/notes/for-home.md` in words with no internal names (no tool hosts, project keys, people or data), so the user can redo it at home. When the redone fix arrives by pull, drop the local commit.
- Anything that would change a tracked file for good (a golden op rename, a DESIGN.md note, a new feature) goes in `for-home.md` the same way.

## Rules for this work

- Verification before every commit on `work-fixes` is `make lint && make test`.
- Paid steps (`make accept`, `oracle eval`) spend Bedrock tokens. Give the estimate before each one and wait for a go. The last full gate at home was about 200K tokens and 7 minutes.
- Never push. Never copy internal data into the repo.
- No new dependencies without asking (CLAUDE.md).

## Phase 1: set up (about 30 minutes)

1. `git remote set-url --push origin DISABLED`, then `git remote -v`: the push URL must read `DISABLED`.
2. `uv sync --all-extras`, then `make lint && make test`. Both must be green before anything else.
3. Set the work models (DESIGN.md, "Tiers and model configuration"): `ORACLE_MODEL_PLANNER`, `ORACLE_MODEL_EXTRACT` and `ORACLE_MODEL_DEEP` as `openai:bedrock/...` strings, with `OPENAI_BASE_URL` pointing at the LiteLLM proxy. Leave `ORACLE_PROFILE` unset and set no `registry` in `~/.oracle/config.toml`, so the registry is `~/.oracle/registry/tools/`; `oracle tool list` prints it.
4. With no tools registered, `oracle ask` reports a gap without calling a model, so the models are proven in phase 2, after the first tool is added.

## Phase 2: add the real tools and recipes (about half a day)

For each of acli, workiq, code search, code graph, kubectl, argocd, es and mongo:

1. `oracle tool add <command>`, review the draft against the real `--help`, accept it.
2. `oracle tool test <tool>` until its examples pass.
3. After the first tool only: ask it one real question with `--level quick --format json`. `meta.model_calls` of 1 or more, and findings with pointers, prove the work models answer.

Then check these things:

- Golden op names survive. `oracle tool test` notes any op the golden cases name that the work registry lacks. Prefer naming the work op to match. If a name truly differs, write it in `for-home.md`; the rename is made at home.
- kubectl cannot return secrets. Its resource-type enum must leave out `secrets` (and `configmaps` if they hold credentials here). Redaction (`exec/redact.py`) is the second line of defence, not the first.
- Auth comes from env vars only (`auth.env` in the manifest), never from argv.
- Each manifest has a `version` block. If a CLI answers neither `--version` nor `version --client`, write the block by hand with the argv that prints its version.

Rebuild the four seeded recipes in `~/.oracle/registry/recipes/`, following `registry/recipes/README.md`; several golden cases check them by name.

Finish with `oracle tool test --ranker`. It must be at or above 90% top-3. Ranker misses on real queries go in `~/.oracle/ranker_eval.yaml` after review.

## Phase 3: acceptance on real data (about 1 hour, paid)

1. Checks on home-fixture data sit in each case's `profiles.home` block. At work such a case runs on its shared checks and fails with "no 'default' block" until it has one. For each: run `oracle eval --case <id>` (paid; about 40K tokens for the crawl case), read the answer in `~/.oracle/eval/last.json`, propose a block with the real pointers and rubric, and wait for a yes. It goes in `~/.oracle/golden.yaml` as `{id: <case>, profiles: {default: {expect: {pointers: [...]}, judge: "..."}}}`.
2. Run `make accept` with the work models, not `make gate-m4`: the gates run the home profile and index the home fixtures. Fix what real data breaks (on `work-fixes`, written up for home); don't loosen a check to pass.
3. Record the result in `~/.oracle/notes/`, and the parts with no internal names in `for-home.md`, so DESIGN.md gets its "as built" note at home.

## Phase 4: Claude Code uses it (about 30 minutes)

1. `uv tool install --editable .` so `oracle` is on PATH.
2. Write the main agent's description of the oracle (DESIGN.md, "CLI interface and output contract"): a few lines on what it searches, that it takes intent rather than keywords, and the three flags that matter. Put it in `~/.claude/CLAUDE.md` or a skill; ask which.
3. Optional: register `oracle-mcp` (the MCP wrapper, built in M3) if MCP suits this Claude Code setup better than Bash.
4. Use it for real work for a week or two. The logs in `~/.oracle/log.sqlite` are what `oracle report`, `oracle dream` and phase 5 read.

## Phase 5: is the oracle worth it? (after about a week of use)

Run the comparison DESIGN.md plans under "Build vs buy": the main agent loading the tool skills directly (and Claude Code's own tool search) against the oracle, on the golden set, measuring accuracy and main-context tokens. Write the verdict and the numbers in `for-home.md`. If the oracle doesn't clearly win, say so: phase 6 adds more to it.

## Phase 6: design features not built yet (built at home)

These are built on the personal laptop against the fakes, each with its gate in DESIGN.md first, and arrive here by pull. What this laptop adds is evidence: from `oracle report`, say in `for-home.md` which items the real logs show matter, in words with no internal names. Today's order:

1. **Caching** (DESIGN.md, "Caching, prior findings"): op results across calls with a per-tool `cache_ttl` in the manifest; plans keyed on the normalised query, intent and ranked op set (7 days, invalidated when a manifest, recipe or prompt changes); and reported findings per session ("see call 12"). Only the pointers table exists today.
2. **Prior findings as a source**: `findings.search`, ranked like any tool, under the three rules in DESIGN.md (a lead with provenance, validity keyed to the source, stored with its Coverage).
3. **Tool drift, the schedule**: manifests already record the binary's version and `oracle tool test` notes a change. Left: run `oracle tool test` weekly as a scheduled task on this laptop (no code needed), and ask how the user wants a drift note delivered.
4. **Open items from "Workflows as tools"**: one budget per trace shared through `log.sqlite`, and `oracle report` drawing the call tree. (Ingestion as a workflow is deliberately a command; confirm before building it.)
5. **Dreaming, the rest of it** (DESIGN.md, "Dreaming"): today `oracle dream` only proposes recipes from repeated crawl chains. Still to build are stages 1 to 7: transcripts into a day table, labelling, the other proposal classes, eval-gated promotion, `dreams/<date>.md`, memory maintenance, and the weekly trend. The code is built at home; it runs here on two or three weeks of work logs, and its output stays in `~/.oracle/`. Transcripts are the one place sensitive content enters an oracle process; keep only the query, the result summary and the next user message.

## Build only when triggered

These are DESIGN.md defaults; don't build them ahead of their trigger. A trigger seen here goes in `for-home.md`.

- OpenAPI and GraphQL importers: when a work tool is only reachable that way.
- `custom-*` script manifests: when a real question needs one.
- Embeddings in the ranker: when `oracle tool test --ranker` drops below 90% top-3.
- A step loop instead of plan-then-execute: if `oracle report` shows more than about 30% of standard calls replanning.
- A deep-tier container: if deep calls exceed a few a day.
- Deferred on 2026-10-25: script-to-recipe promotion, choosing between crawl and script automatically, an OS-level sandbox, and architecture candidates 4 to 6 (one summary chain, PlanRunner owning step identity, the log store owning its queries).
