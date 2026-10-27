# oracle

One read-only CLI that a main agent (Claude Code) calls to find things across every data source
the user has. It takes a question in plain language, picks the right tools, runs them in an
isolated context, and returns distilled findings with pointers back to the raw data. The main
agent never loads the tool definitions or sees the intermediate results.

Status (2026-10-26): milestones M1 to M4 are built and `make gate-m4` is green at home: 317
tests, 26/26 manifest examples, ranker top-1 89% and top-3 97%, golden set 7/7 at about 29K
sub-agent tokens and 27 s per case. Why it is built this way, and what each part does, is in
[docs/DESIGN.md](docs/DESIGN.md).

## Quick start (personal laptop)

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). `make test` needs nothing else. A real
call at home also needs:

- Ollama, the `vex` CLI and `codebase-memory-mcp`, for `make fixtures` (it indexes the fixture
  code repo, `tests/fakes/acme/`)
- the Claude Code CLI, for the `claude-code:*` models (the Claude subscription)

```sh
uv sync --all-extras
make test
make fixtures

export PYDANTIC_AI_NO_BANNER=1 ORACLE_PROFILE=home
export ORACLE_MODEL_PLANNER=claude-code:sonnet ORACLE_MODEL_EXTRACT=claude-code:haiku \
       ORACLE_MODEL_DEEP=claude-code:opus
uv run oracle "who is andrew neville"
```

The home profile runs against fakes: acli, workiq, kubectl, argocd, es and mongo are stdlib
CLIs over fixtures in `tests/fakes/`, and code search and the code graph see only the fixture
repo. `uv tool install --editable .` puts `oracle` on PATH.

## Using it

```
oracle "<question>" [--intent "<why you are asking>"] [--level quick|standard|deep]
       [--scope <tool[,tool]>] [--prefer <tool>] [--breadth] [--format md|json] [--session <id>]
       [--depth <hops>] [--budget <tokens>|unlimited]   # deep crawl only
       [--script]                                       # deep: one script instead of a crawl
```

`oracle "<question>"` is short for `oracle ask "<question>"`. The level defaults to `standard`
(plan, run, completeness check, distill). `quick` is one op. `deep` crawls what the results
mention, or with `--script` writes one script against the shortlisted tools.

Every answer has the same four sections, in this order:

```
## Findings   3-15 short items, each with a pointer (code:path:line, jira:KEY, conf:ID, ...)
## Coverage   what was searched and what was not, confidence, gaps
## Refine     1-3 ready-to-run variants
## Trace      one line per op, and the call id
```

Other commands:

| Command | What it does |
| --- | --- |
| `oracle expand <pointer>` | The full record behind a pointer. |
| `oracle run <tool.op> key=value ...` | One op or workflow directly, JSON out. |
| `oracle feedback <call_id> --useful yes\|no` | Record whether an answer helped. |
| `oracle eval [--case <id>]` | Run `golden.yaml`. Spends model calls. |
| `oracle report` | Signals from the log: re-asks, used findings, replans, ranker misses. |
| `oracle dream [--accept <name>]` | Propose recipes from crawl chains that repeat in the log; `--accept` installs one. |
| `oracle tool list\|test\|add\|activate\|build` | Registry management. `test --ranker` runs the ranker eval; `add` spends 5 to 30 model calls. |

`oracle-mcp` (the `mcp` extra) exposes the same thing as one MCP tool.

## Configuration

`~/.oracle/config.toml` holds defaults (copy [config.example.toml](config.example.toml)), and
environment variables override it. `ORACLE_HOME` moves `~/.oracle`.

- **Models.** `ORACLE_MODEL_PLANNER`, `ORACLE_MODEL_EXTRACT` and `ORACLE_MODEL_DEEP` take
  PydanticAI model strings: `claude-code:<alias>` at home, `openai:bedrock/...` through a LiteLLM
  proxy (`OPENAI_BASE_URL`) at work. No model name is hard-coded.
- **Profile.** `ORACLE_PROFILE` (or `profile =`) selects `profiles/<name>.toml` in this repo, or
  `~/.oracle/profiles/<name>.toml`, which wins. A profile sets the registry, PATH additions and
  env for tool subprocesses.
- **Registry.** `ORACLE_REGISTRY`, then the profile's `registry`, then `config.toml`, then
  `~/.oracle/registry/tools/`. Recipes live beside it in `recipes/`.
- **Local eval data.** `~/.oracle/golden.yaml` and `~/.oracle/ranker_eval.yaml` are read on top
  of the repo's files. The local golden file can add cases and `profiles.<name>` blocks to repo
  cases by id, but cannot change a repo case.

Logs, caches, staged ingestions and the generated `oracle_tools` library live under
`~/.oracle/`. `oracle tool activate` and `oracle dream --accept` write into the active registry
and its recipes, which at home is `profiles/home/` in this repo.

## Two laptops

The oracle is built on a personal laptop with fakes and run on a work laptop with the real
tools. At work, each real tool is added with `oracle tool add` into `~/.oracle/registry/tools/`,
and everything that names internal tools or data stays under `~/.oracle/`. Code goes one way:
this repo is pushed from home and pulled at work, never pushed from there.
[docs/WORK-LAPTOP.md](docs/WORK-LAPTOP.md) is the phase-by-phase plan for the work side.

## Developing

```sh
make lint    # ruff, pyright, xenon, radon and vulture
make test    # unit tests, no models, no tools
make build   # uv sync, then a wheel and sdist in dist/
```

The gates spend model calls; the last full `make gate-m4` was about 200K tokens and 7 minutes.

| Target | Checks |
| --- | --- |
| `make gate-m1` | Every manifest example passes against the home profile. |
| `make gate-m2` | The golden set passes against the home profile. |
| `make gate-m3` | M2, plus the ranker eval at 90% top-3 or better. |
| `make gate-m4` | M3, plus a recipe learned by `oracle dream` from the logs. |
| `make accept` | The same checks on the active registry, with no profile (for the work laptop). |

`golden.yaml` is the acceptance set: when a case fails, the fix goes into the oracle (ranker
text, recipes, prompts, fixtures), not into the case. See [CLAUDE.md](CLAUDE.md) for the rules
coding agents follow here.

## Where things are

| Path | What |
| --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | The design: decisions, manifest format, tiers, module layout, milestones. |
| [docs/WORK-LAPTOP.md](docs/WORK-LAPTOP.md) | Finishing and running the oracle on the work laptop. |
| `oracle/` | The package. `ask.py` is one call end to end; `strategy.py` holds the four tiers. |
| `profiles/home/` | The home registry (`tools/`) and recipes. |
| [registry/README.md](registry/README.md) | Where the work registry lives, and how to rebuild the seeded recipes. |
| `skills/` | Skills the oracle's own agents load (JQL, CQL, people lookup, ...). |
| `golden.yaml`, `ranker_eval.yaml` | The acceptance set and the ranker eval. |
| `tests/fakes/` | Fake CLIs, their fixtures, and the acme fixture code repo. |
| `adapters/bin/` | Thin JSON CLIs over local tools (`cmm-graph` over codebase-memory-mcp). |
