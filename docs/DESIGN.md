# Oracle: Tool Routing Design

Oct 24, 2026 · @Andrew

## Purpose and priorities

The oracle is one CLI tool that a main agent (Claude Code) calls repeatedly to find things across every data source the user has, without the main agent knowing which tools exist or how to call them. It takes a question in plain language, picks the right tools, runs them, and returns distilled findings with pointers back to the raw data.

Priorities, in order, and what each one decides:

1. **Accuracy and completeness.** The oracle verifies its own coverage before returning, fans out across every applicable source for discovery questions, and reports what it did not search. It uses Sonnet-class models for planning, not Haiku.
2. **Token efficiency and cost.** Tokens in the main agent's context are the expensive ones: they are re-read on every later turn and degrade the agent making the important decisions. The oracle spends its own sub-agent tokens freely to keep the main context lean. Intermediate data never reaches the main agent.
3. **Time efficiency.** Parallel execution everywhere it is possible, but never at the expense of 1 or 2. A completeness pass that adds two seconds is kept.

The division of judgment: **the oracle decides how to answer the question completely; the main agent decides what question to ask next.** Reading a result and choosing a new direction belongs to the main agent, which has the user's intent and conversation history. Noticing that the first result is incomplete and running one more step belongs to the oracle.

The oracle is read-only. Anything that writes, sends or deletes stays a direct tool the main agent calls deliberately.

## Key decisions

Each decision names the alternative it rejects and the evidence behind it.

| Decision | Rejected alternative | Why |
| --- | --- | --- |
| One `oracle` CLI tool, described in CLAUDE.md or a skill | Load every tool definition into the main agent; or load skills on demand | Tool selection accuracy degrades past 30–50 tools and token overhead reaches 100K+ (Anthropic). A tool definition is re-presented every turn; a skill loaded at turn 3 is buried by turn 60. |
| Tools registered as declarative **ops manifests** (typed params, argv templates, returns schema) | Register raw `--help` docs and let the model write shell commands | Function calling with schemas has the best measured reliability; shell-string composition is the widest attack surface and has no output contract, so ID chaining is guesswork. |
| Manifests are the source of truth; a generated Python library is derived from them | Two hand-maintained formats | One authored artifact, two executors. |
| Standard tier uses **plan-then-execute** with bounded replan | Open-ended agentic step loop | Dependent call chains score \~28% for GPT-4o (NESTFUL). Plan-and-execute systems (ReWOO, LLMCompiler) reported several-fold token and latency savings over ReAct at equal or better accuracy, and parallel fan-out comes for free. |
| Recipes (pre-built composite ops) for every chain seen more than a few times | Let the planner improvise every chain | Moves chain reliability into deterministic, testable code. |
| **Completeness pass** after execution | Return first results | Self-reflection was one of two components that mattered in AnyTool's ablations. Accuracy is priority 1. |
| Rank **ops** grouped by tool, hybrid: exact name → BM25 → embeddings; replans condition on prior results | Rank tools only, semantic only, query only | Hybrid beats pure semantic because tool docs contain exact terms; conditioned retrieval beats query-only on multi-step benchmarks; hierarchical tool→op routing is what MCP-Zero and AnyTool converged on. |
| Sonnet plans; Haiku only extracts and formats | Haiku everywhere for cost | Anthropic's own tool search does not support Haiku. Planning is where accuracy is decided. |
| Deep tier is a hop-based crawl by default, and writes a script against the generated library when the chain needs code logic | Deep tier as a longer step loop; a recursive `oracle` call per extracted term | Code execution measured \~77% token and \~70% cost reduction at equal success, and chains and filtering are native in code. A crawl must read results to find terms of unknown kinds, which a script written up front cannot do; batching a whole hop keeps model calls per hop constant however many terms it has. |
| Read-only, enforced at the manifest level | Allow mutating ops behind a flag | Read-only sub-agents are the one multi-agent pattern both Anthropic and Cognition endorse. |
| PydanticAI, model via config string | Claude Agent SDK | Work environment is Bedrock through LiteLLM; the oracle is a small typed agent, not a coding agent; structured output validation is the core need. |
| Python, asyncio | Go | The workload is I/O-bound shell and HTTP calls; the GIL is irrelevant. Revisit only if a real sandbox or thousand-way fan-out appears. |

## Build vs buy

No existing product does what the oracle does end to end: rank across heterogeneous local CLIs and APIs, execute in an isolated context, chain deterministically, check completeness, and return distilled findings with coverage. Several products solve one slice, and three of them belong *inside* the oracle as registered tools rather than instead of it.

| Candidate | What it solves | What it leaves | Verdict |
| --- | --- | --- | --- |
| Anthropic Tool Search Tool (API) and Claude Code's built-in MCP tool search | Tool-definition bloat in the main agent; measured accuracy gains at 30+ tools. | Results still land in the main context; no chaining, completeness or coverage; Claude-API feature, so availability through a LiteLLM/Bedrock proxy must be verified. | The real alternative. Already planned as the A/B: skill files loadable directly by the main agent, compared against the oracle on the golden set. |
| MCP aggregators: MetaMCP, ToolHive meta-mcp, Docker MCP Gateway | One endpoint fronting many MCP servers; hybrid find\_tool / call\_tool discovery; some add code-mode. | Only for tools that are MCP servers; selection by the main agent, results in its context; no manifests for local CLIs, no plan DAG, no distill. | Not a replacement. Possibly a transport for MCP-type tools inside the registry if the user already runs one. |
| Composio Tool Router | Dynamic tool loading plus managed auth for hundreds of SaaS apps through one MCP endpoint. | Built for SaaS connectors with hosted auth; the user's tools are local CLIs, internal APIs and env-var auth; data goes through a third party. | No. Auth is not the problem being solved. |
| langgraph-bigtool | Semantic retrieval over a large tool registry inside a LangGraph agent. | Retrieval only; brings the LangGraph dependency; main-agent context still holds results. | No. The ranker is \~200 lines of BM25 plus RRF; the dependency is larger than the feature. |
| Glean (remote MCP server: search, chat, read\_document, people search, code search) | One index over Confluence, Jira, GitHub, Teams and more with ACLs; strong people and document answers. | Requires a Glean license at work; does not cover the code graph, ES, Mongo, kubectl or argocd; results are its own summaries. | Buy as a *source* if licensed: register as a tool and let the ranker weigh it against acli and workiq. It does not replace the orchestrator. |
| Atlassian's own MCP server (Rovo) for Jira and Confluence | Maintained schemas and auth for the two Atlassian tools. | Covers two of nine tools. | Prefer over hand-wrapping `acli` if available at work; register through the MCP import path. |
| M365 Copilot / Work IQ | Teams, mail and people in one index. | `workiq` is already that surface as a CLI. | Already in use; nothing to add. |

What this comparison changes: the ingestion workflow's MCP import path moves up in priority (Atlassian, possibly Glean), and the A/B against native tool search is a named deliverable rather than an afterthought. What it does not change: the orchestrator is built, because the two properties the priorities depend on, isolated execution and coverage reporting, are not sold by anyone.

## Architecture

&#91;embedded content: oracle request flow · 5 steps, 1 replan loop\]

One call runs five steps. The ranker shortlists ops from the registry; Sonnet turns the question into a plan DAG (or a script at the deep tier); the executor runs it with parallelism, pagination and output caps; the completeness check looks for gaps and replans at most twice; Haiku distills findings with pointers. Only the distilled result crosses back to the main agent. Every op and model call is logged.

The registry is one directory of manifests. A codegen step derives a typed Python library from it for the deep tier, so there is one authored artifact and two executors.

## CLI interface and output contract

The main agent sees one command. Its description in CLAUDE.md or the skill is a few lines: what it searches, that it takes intent not keywords, and the three flags that matter.

```
oracle "<question>" [--intent "<why you are asking>"] [--level quick|standard|deep]
       [--scope <tool[,tool]>] [--prefer <tool>] [--breadth] [--raw <pointer>]
       [--format md|json] [--session <id>]
       [--depth <hops>] [--budget <tokens>|unlimited]   # deep crawl only
oracle expand <pointer>        # fetch the full record behind a pointer
oracle feedback <call_id> --useful yes|no [--note "..."]
oracle tool add|test|list      # registry management
```

- `--intent` is optional but changes results: "where ownership transfer happens, I'm tracing a bug" ranks differently from the bare question.
- `--level` defaults to `standard`. `quick` is one op, no completeness pass. `deep` enables the crawl (default for discovery questions) and the script runner.
- `--depth` and `--budget` bound a deep crawl: hops (default 3) and model tokens (default `ORACLE_DEEP_BUDGET`; `unlimited` leaves depth as the only limit).
- `--breadth` forces fan-out across every applicable tool; the planner also turns it on by itself for discovery-shaped questions ("everything about", "who is", "what happened").
- `--session` lets `expand` resolve pointers from earlier calls; Claude Code passes a stable id per conversation.
- `--format md` is the default for Bash callers; `json` is for the optional MCP wrapper.

**Output contract.** Every response has these sections, in this order, so the main agent can make its cross-question judgment:

```
## Findings
<3–15 short items, each with a pointer: code:path:line, jira:KEY, es:index/id, teams:msg-id>

## Coverage
Searched: code-graph, semantic-search, confluence   Not searched: jira, es, workiq
Confidence: high|medium|low   Gaps: <what the question asked that no result covers>

## Refine
oracle "..." --scope jira        # 1–3 ready-to-run variants

## Trace
<op> <args> → <n results, ms>   (one line per op; call_id at the end)
```

Findings are distilled, never raw. Pointers are stable ids the main agent can hand back to `expand`. Coverage is mandatory: an answer without "Not searched" invites the main agent to assume completeness it does not have. The whole response targets under 1,500 tokens at `standard`; the trace is the only section the main agent may ignore.

**MCP wrapper (milestone 3).** A thin MCP server that shells to the CLI with `--format json` and exposes one `oracle` tool. Nothing in the core depends on it.

## Tool registry

A tool is one YAML file in `registry/tools/<name>.yaml` exposing a small set of named ops. Everything the planner sees, the executor validates, and the codegen emits comes from this file.

```yaml
name: code-search
description: Semantic search over local code embeddings (sqlite). Finds symbols and files by meaning.
tags: [code, search, symbols, semantic]
when: "Locate code by what it does, when the exact name is unknown."
not_when: "Exact symbol name known — use code-graph.find_symbol instead."
auth: {env: [CODE_INDEX_PATH]}
limits: {timeout_s: 20, concurrency: 4, max_output_kb: 256}
mutating: false
ops:
  - name: search
    when: "Find code related to a concept or behaviour."
    params:
      query: {type: str, required: true, desc: "natural language, 3–12 words"}
      repo:  {type: str, enum_from: code-graph.list_repos}
      limit: {type: int, default: 20, max: 100}
    argv: ["csearch", "--json", "-n", "{limit}", "{?repo:--repo,repo}", "{query}"]
    returns:
      format: json
      path: "$.results[*]"
      id_field: symbol_id
      summary_fields: [path, line, name, score]
      pointer: "code:{path}:{line}"
    chains_to: [code-graph.expand]
    examples:
      - params: {query: "ownership transfer", limit: 10}
        expect: {min_results: 1, has_fields: [path, line]}
```

Field by field, and the corner case each exists for:

| Field | Purpose | Corner case it handles |
| --- | --- | --- |
| `argv` as a list | The executor never builds a shell string | Quoting and injection. `{?param:--flag,param}` emits the flag pair only when the param is set; `{?param:--flag=,param}` emits one `--flag=value` item, for values such as `-7d`. `{stdin:param}` feeds a param on stdin. A value that would stand alone and starts with `-` is refused unless a literal `--` came earlier, so a param cannot smuggle in an undeclared flag. |
| `returns.format` | json, jsonl, table, text | CLIs without `--json` get `table` (a column parser) or `text` (truncated, handed to the extractor). |
| `returns.id_field`, `pointer` | Deterministic chaining and stable pointers | Step two binds to `$1.symbol_id`; the main agent gets `code:path:line` it can `expand`. |
| `paginate` | `{param: cursor, next: "$.next_cursor", max_pages: 5}` | The executor pages; the planner never sees pagination. |
| `auth.env` | Names of required variables | Checked before running: a missing variable is a clean "not configured" line in Coverage, not a stack trace. Secrets never pass through a model. |
| `limits` | Timeout, concurrency, output cap | A 10 MB log result is written to a temp file and returned as a pointer. Per-tool concurrency respects rate limits. |
| `mutating` | Read-only enforcement | The oracle refuses to load a mutating op at any tier. |
| `when` / `not_when` | Selection text | Overlapping tools (code-search vs code-graph, es vs mongo) are the top documented cause of wrong picks; this is where the boundary is written. |
| `enum_from` | Values that come from another op | The executor resolves it (cached per session) so the planner picks from real repo names. |
| `async` | `{submit: op, poll: op, done: "$.status == 'complete'"}` | Long-running jobs (ES async search, large k8s queries). |
| `examples` | Real invocations | Doubles as the smoke test for `oracle tool test`. |
| `escape` | `{argv: ["kubectl"], allow: deep}` | Raw arguments for the long tail, deep tier only, logged loudly, still subject to a read-only subcommand allowlist. |

**Import paths.** `oracle tool add` drafts a manifest from the best available source and then runs the examples:

- OpenAPI spec → one op per selected endpoint (the user picks which to expose; never all).
- MCP server → ops from its tool schemas, argv replaced by an MCP transport block.
- GraphQL → one op per named query the user registers.
- Elasticsearch, Mongo, graph DB → parameterised read-only queries as ops; a `free_query` op gated to deep tier on a read-only connection.
- CLI → Sonnet drafts ops from `--help` and 2–3 invocations from shell history; the user edits; examples must pass before the tool is active.

**Not supported directly.** Interactive and TUI tools (k9s), REPL sessions, tools that prompt. Wrap in a script that exposes a clean op and register the script. `kubectl` covers what k9s would.

**Versioning.** Manifests live in git. A manifest change re-runs that tool's examples and the golden set.

## Worked manifests for the current tools

The registry starts with these. Op names are proposals; the real flags come from each tool's `--help` during `oracle tool add`.

| Tool | Type | Ops (v1) | Returns | Notes |
| --- | --- | --- | --- | --- |
| `code-search` | CLI over sqlite embeddings | `search` | json, `code:{path}:{line}` | Chains to `code-graph.expand`. |
| `code-graph` | graph DB of code | `find_symbol`, `expand` (callers, callees, refs by id list), `symbols_at` (path:line → enclosing symbol), `path_between`, `list_repos`, `free_query` (deep only, read-only) | json, `code:{path}:{line}` | `expand` takes `ids: list[str]` so one op serves a whole result set. `symbols_at` bridges search hits to graph ids when the search tool does not return graph symbol ids (true at home: vex returns chunk ids). |
| `acli` | Atlassian CLI | `jira.search` (JQL), `jira.get`, `confluence.search` (CQL), `confluence.get` | json, `jira:{key}`, `conf:{id}` | Two `when` lines: tickets and status vs documents and decisions. |
| `workiq` | M365 CLI | `people.lookup`, `teams.search`, `mail.search`, `calendar.today` | json, `teams:{msg-id}`, `mail:{id}` | `people.lookup` is the primary source for "who is X". |
| `kubectl` | CLI | `get` (resource, namespace, selector), `events` (namespace, since), `logs` (pod, since, grep) | json via `-o json`; logs are text, capped | Read-only subcommand allowlist on `escape`. |
| `argocd` | CLI | `app.list`, `app.get`, `app.history` | json | Deploy events feed "what happened today". |
| `es` | Elasticsearch | `search` (index, query\_string, since, size), `free_query` (deep) | json, `es:{index}/{id}` | Index list via `enum_from: es.list_indices`. |
| `mongo` | Mongo search | `find` (collection, filter, projection, limit) | json, `mongo:{collection}/{id}` | Filter is a JSON param validated as an object; no `$where`. |
| `custom-*` | Python/Go scripts to MCP and REST | one op per script entry point | json | Scripts already speak JSON; manifest is argv plus returns. |

Recipes to seed, each a composite op:

- `code.find_and_expand` = `code-search.search` → `code-graph.expand($1.ids)` → merge. Covers "where does X happen in code". If the search tool's ids are not graph symbol ids, insert `code-graph.symbols_at($1 path:line)` before the expand; verify which applies to `csearch` on the work laptop.
- `code.trace_path` = `code-graph.find_symbol(a)` ∥ `code-graph.find_symbol(b)` → `code-graph.path_between`. Covers "how A talks to B".
- `people.profile` = `workiq.people.lookup` ∥ `acli.jira.search(assignee)` ∥ `workiq.teams.search(from)` → merge. Covers "who is X".
- `today.events` = `workiq.calendar.today` ∥ `acli.jira.search(updated ≥ today)` ∥ `argocd.app.history(today)` ∥ `kubectl.events(since 24h)` ∥ `workiq.teams.search(since today)` → rank by the extractor. Covers "important events from today".

These four recipes cover the four seed queries. The planner reaches for a recipe before composing from primitives.

## Ranking

The ranker turns the question into a shortlist of ops the planner is allowed to use. It is deterministic Python with no model call, so it is fast, testable and the first place to look when a wrong tool is picked.

Index unit: one document per **op**, text = tool name + op name + tool description + `when` + param names and descriptions + example queries. Each document carries its parent tool, so results are grouped by tool and the planner sees the whole op list of each selected tool (hierarchical tool→op, as MCP-Zero and AnyTool do).

Stages, in order, scores combined by reciprocal rank fusion:

1. **Exact name.** A tool or op name appearing in the query or `--scope` pins that tool to the top. `--prefer` adds a fixed bonus.
2. **BM25** over the op documents (`rank_bm25`).
3. **Embeddings** over the same documents, local model (`sentence-transformers`, cached vectors rebuilt on manifest change). Optional in v1; enable once the registry passes \~40 ops or BM25 misses show up in the logs.
4. **Recipes first.** A recipe whose `when` matches outranks its own primitives.

Output: top 3–5 tools with all their ops, which is the range Anthropic's tool search returns and the size that keeps the planner accurate. With the current registry (9 tools, \~30 ops) the ranker mostly prunes; it matters more as scripts get added.

**Conditioned retrieval on replan.** When the completeness check sends the planner back, the ranker runs again with the original query plus the gap description plus the summary fields of what was found. A result mentioning a Jira key re-ranks `acli` up even if the first query never mentioned tickets.

**Breadth mode.** Everything above a score threshold is returned, not a top-k, and the planner fans out across all of it.

**Evaluation.** `ranker_eval.yaml` holds query → expected tools. `oracle tool test --ranker` reports top-1 and top-3 hit rates. Manifest `when` text is tuned against this file, not by intuition.

## Planner and executor

The standard tier makes two or three model calls per oracle invocation: one to plan, one to distill, and sometimes one to check. Everything between them is deterministic.

**Plan format.** The planner (Sonnet, PydanticAI with a `Plan` output model) emits a small DAG. Validation rejects unknown ops, bad params, and references to steps that do not exist.

```json
{
  "interpretation": "Find code where ownership of a resource is handed from one owner to another",
  "steps": [
    {"id": "s1", "op": "code-search.search", "params": {"query": "ownership transfer handoff owner change", "limit": 20}},
    {"id": "s2", "op": "code-graph.find_symbol", "params": {"pattern": "*transfer*owner*"}},
    {"id": "s3", "op": "code-graph.expand", "params": {"ids": "$s1.ids + $s2.ids", "direction": "callers"}, "after": ["s1", "s2"]},
    {"id": "s4", "op": "acli.confluence.search", "params": {"cql": "text ~ 'ownership transfer'"}}
  ],
  "breadth": false,
  "expect": "file:line locations plus the design doc if one exists"
}
```

Bindings: `$s1.ids` is the `id_field` of every result of s1; `$s1.pointers` its pointers; `$s1.results[0].path` reaches into a record (one value; `[*]` gives a list); `+` concatenates lists. A binding is a whole param value. Steps with no `after` and no bindings run in parallel, in dependency waves. A step bound to a step that returned nothing is skipped and says why. An identical `(op, params)` already run in the call is reused, not run again. `expect` is what the completeness check reads.

**Executor.** Pure async Python. For each step: resolve bindings, validate params against the manifest (pydantic), check `auth.env`, build argv, run with timeout under the tool's concurrency semaphore, parse per `returns`, page if `paginate`, cap output, attach pointers. A failed step records a structured error and does not stop siblings. Results are stored in a per-session SQLite cache keyed by pointer, which is what `oracle expand` reads.

**Recipes** are ops whose implementation is a stored plan with free parameters. The planner sees them as ordinary ops with a `when`, listed before the tools. A recipe can declare `on_empty: fallback_plan` for the common case where the first stage returns nothing. Recipes live beside the registry they use (`<registry>/../recipes/`, so `profiles/home/recipes/` at home and `registry/recipes/` at work), because a recipe names ops, and the two machines have different ops. A stage's params may use the recipe's params (`{concept}`), the resolved time window (`{window_start}`, `{window_end}`) and sibling stages (`$search.pointers`); stages run as `s1/search`, and a binding to the recipe step reaches its `output` stages. A stage whose tool is not active here does not fail the load: it runs as "not available here", which is what Coverage reports.

**Completeness check.** After execution, one Sonnet call sees the plan's `expect`, the question, and the summary fields of every result (not raw bodies). It returns `{"complete": bool, "gaps": [...], "next_steps": [<plan steps>] | null}`. If `next_steps` is non-empty and the replan count is under 2, the executor runs them and the check runs once more. Skipped at `quick`. The check is the mechanism for within-question judgment: a synonym retry, a second naming convention, a tool that returned nothing and has a sibling.

**Breadth mode.** Plan = one step per applicable tool, same query adapted per op, all parallel. The distill step groups by source and states where sources agree, disagree or are silent, rather than flattening into one list.

**Distill.** Haiku, with a `Findings` output model: items with pointers, coverage, refinements. It gets summary fields plus a capped slice of each result, never the full payloads. If a result's text was truncated, the finding says so and the pointer is how the main agent gets the rest.

**Bounds that keep accuracy-first from becoming search-everything:** max 12 steps per plan, 2 replans, 60 s wall clock at standard, 5 pages per paginated op, and a `sufficient` field in the check that short-circuits when the question is plainly answered. As built in M2: no new check starts after 60 s; every model call is capped at 90 s (a `claude-code` call at 120 s inside the wrapper, so a hung process can never hang the main agent); past 90 s distill is skipped and findings are raw summary fields, with a note.

## Tiers and model configuration

A tier is a ceiling, not a target: a `deep` call that resolves in one step stops after one step.

| Tier | Planner | Distill | Steps | Replans | Script runner | Use when |
| --- | --- | --- | --- | --- | --- | --- |
| quick | Sonnet | Haiku | 1 | 0 | no | One known lookup: "get ticket PROJ-123". |
| standard (default) | Sonnet | Haiku | 12 | 2 | no | Everything else, including the four seed queries. |
| deep | Opus (script) / Sonnet (crawl hops) | Sonnet | crawl: hop depth 3 within `--budget`; or one script | per hop | yes | Discovery that follows what results mention (crawl), or multi-source tracing with judgment inside the question (script), or when standard reports low confidence. |

The oracle escalates one tier by itself when it hits the cap without a `complete` check, says so in the trace, and never escalates into `deep` without `--level deep` or `ORACLE_AUTO_DEEP=1`, because deep calls can run a minute and cost real money. The same holds for budget: `--budget unlimited` is only ever set by the caller or by `ORACLE_DEEP_BUDGET`, never chosen by the oracle.

**Model configuration.** Every model is a PydanticAI model string from the environment, so the same binary runs at home and at work:

```
ORACLE_MODEL_PLANNER=claude-code:sonnet                    # home, Claude subscription via Claude Code
ORACLE_MODEL_PLANNER=anthropic:claude-sonnet-...          # home, direct API key
ORACLE_MODEL_PLANNER=openai:bedrock/anthropic.claude-...  # work, via LiteLLM
OPENAI_BASE_URL=https://litellm.internal/v1               # LiteLLM proxy
ORACLE_MODEL_EXTRACT=...   ORACLE_MODEL_DEEP=...
```

PydanticAI's OpenAI-compatible provider targets the LiteLLM proxy; its Bedrock provider is the fallback if the proxy is unavailable. `claude-code:<alias>` is the oracle's own PydanticAI model (`oracle/llm/claude_code.py`): each request is one Claude Code process through `claude-agent-sdk` with no tools, no MCP servers and no settings, and the agent's output schema goes to the CLI's structured output, so validation and retries stay PydanticAI's. It is the same route rufalo's `agent_sdk` provider uses, works when the oracle is itself called from inside Claude Code, and adds roughly 1 s of process start per model call. Model names are never hard-coded. One config file `~/.oracle/config.toml` holds defaults; environment variables override it.

**Cost expectations at standard.** Two model calls (plan, distill) plus a check: roughly 8–15K input tokens and 1–2K output per invocation, almost all of it in the sub-agent. The main agent receives under 1.5K. That ratio is the point.

## Deep tier: crawl

The crawl is the default deep strategy for discovery-shaped questions: search, read what the results mention, search those, and repeat. "What is going on with feature X" finds a design doc, which names people and a Jira key; the ticket has an epic; the epic links a PDD; the PDD names more people. No list of entity types is declared anywhere, because no list could cover every tool: a model reads each hop's results and says what is worth searching next, in plain words.

**One hop.**

1. **Extract** (Haiku). Reads the summary fields and capped slices of the hop's new results, inside the delimited data block, under the same prompt-injection rule as distill. Returns terms, each with a value, a plain-words description and a relevance score for the question: `PLAT-412` / "Jira ticket the design doc links to" / 0.9; `Jon` / "first name only; reviewed the gateway runbook, OPS space" / 0.6.
2. **Plan** (Sonnet, one call for the whole hop). The terms' descriptions are the ranker queries ("person Jon, OPS space" ranks workiq first); the planner turns all admitted terms into one batch of steps. Splitting extract from plan keeps "Sonnet plans; Haiku only extracts" true.
3. **Validate.** Every step goes through `plan_errors` with a retry, exactly as in the standard tier. Terms come from tool output and are untrusted, so text planted in a Confluence page cannot steer the crawl outside the registry or past read-only.
4. **Run** in parallel under per-tool concurrency. A normalised `(op, params)` already run in this crawl is skipped, which is also what makes a crawl converge.

Stop when the hop depth is reached (default 3, `--depth`), when a hop yields no new terms, or when the budget is spent. Hop depth counts hops inside one invocation; it is not the recursion guard's call-tree depth.

**Budget.** Tool calls cost no model tokens; extraction, planning and the summary do. Before each hop the crawl estimates its cost from the result sizes measured so far, keeps a reserve for the summary, and admits every new term if they fit. Only when they do not are terms admitted in relevance order until the budget is used, so there is no fixed top-k. `--budget <tokens>` sets it per call and `ORACLE_DEEP_BUDGET` sets the default. `--budget unlimited` leaves hop depth as the only limit and still keeps dedupe, per-tool concurrency (pacing, not a cap), a Ctrl-C that returns a summary of everything found so far, and a final cost line (tokens, searches, wall time). At home the crawl shares the Claude subscription with the main Claude Code session, so an unlimited crawl can spend the usage window the main session needs; at work it is Bedrock spend.

**Summary.** Map then reduce: Haiku summarises each result in parallel, one Sonnet call merges them into the output contract. The contract does not change (3–15 findings with pointers); Coverage adds hops run, searches run, terms found but not expanded (with why: budget or depth) and the cost.

**Why not a recursive `oracle` call per term.** The design allows nested calls (workflows use `ctx.oracle.ask()`, subprocesses carry `ORACLE_TRACE`), and recursion would reuse the whole pipeline. As the crawl's inner loop it costs more than it saves: every child runs its own plan, check and distill, so model calls grow with the number of terms instead of the number of hops; each child pays process start (and, at home, a Claude Code process of 0.2–0.6 GB, so a 50-term hop is 50 of them); siblings need a shared seen-set anyway to avoid re-searching each other's terms; and the recursion guard (depth 3, tier clamped one lower per level) would quietly turn hop 2 into a standard call and hop 3 into a quick one. Recursion stays the right tool when one term is itself a question that needs its own investigation: the extractor may mark a term `investigate`, and that term alone becomes a nested `oracle` call under the guard.

**SDK.** The crawl is about 30 lines on a small `oracle.sdk` that model-written scripts and workflows use too: `extract(results)`, `fan_out(steps)`, `summarize(results)`, `crawl(question, depth, budget)` and the shared `budget`, on top of the generated per-op library below. A script that needs custom logic calls `crawl()` or its parts instead of re-implementing them, which keeps scripts short and their mistakes few. smolagents' CodeAgent and Cloudflare's Code Mode are the precedent for models writing code against a tool API; this stays a module inside the oracle, not a framework.

**Logging for dreaming.** Each op row records its hop and the term or pointer that led to it, so dreaming sees chains as sequences of op refs (`confluence.search → jira.get → jira.search(epic) → confluence.get`) with no entity types involved. A chain seen in three or more successful crawls becomes a recipe candidate, as for scripts.

**Proposed acceptance case (for the M4 gate; golden cases are the user's call).** The home fixtures already hold a three-hop chain: the Teams message "Ingestion gateway 5xx again during the Salesforce batch, I took OPS-1206" → `jira:OPS-1206` → the Confluence runbook "Runbook: ingestion gateway 5xx" → its author Priya Raman → `workiq.people.lookup`. Query: "what is going on with the ingestion gateway 5xx errors", at `deep`. Passes when the findings include the ticket, the runbook and Priya's role, each with a pointer, and Coverage lists the hops.

## Deep tier: generated library and script runner

Scripts are for questions where the chain itself needs judgment at each hop that a fixed crawl cannot express: "trace how a request from the API reaches the token store" across the graph DB and semantic search. An Opus planner writes a short Python script against a typed library instead of a DAG, so loops, conditionals and filtering happen in code and, on this path, no intermediate data reaches a model except through the SDK's `extract` and `summarize` calls.

**Generated library.** `oracle tool build` reads every manifest and emits `oracle_tools/<tool>.py`: one typed async function per op, with the manifest's docstring, pydantic-validated arguments, the same executor underneath (timeouts, pagination, caps, pointers), and `mutating` ops simply absent. The library is regenerated on manifest change and never edited by hand.

```python
from oracle_tools import code_search, code_graph

hits = await code_search.search("ownership transfer", limit=30)
frontier = {h.symbol_id for h in hits}
seen, path = set(), []
for _ in range(6):
    callers = await code_graph.expand(ids=list(frontier - seen), direction="callers")
    seen |= frontier
    frontier = {c.symbol_id for c in callers if "api" in c.path}
    path.extend(callers)
    if not frontier:
        break
report(path)  # structured result the distill step reads
```

**Script runner, v1 isolation.** The threat model is accidental damage by a model-written script, not an adversary: the user is the only caller and every tool is read-only. V1 isolation is therefore a subprocess with: a fresh temp working directory; an environment allowlist (only the variables the manifests name); `resource` limits on CPU seconds and memory; a wall-clock timeout; stdout captured and capped; and the generated library as the only import the prompt offers. The script cannot call `subprocess` or open sockets by convention, enforced by an AST check that rejects `subprocess`, `os.system`, `socket`, `requests` and friends before running. This is enough to stop a bad script from doing anything the ops themselves could not do.

**What this does not do.** It does not stop a script from reading arbitrary files on the user's machine or exhausting the network through the ops. A container (Docker, or a lightweight sandbox) is milestone 4 and is only worth building if deep-tier calls become frequent, which the logs will show. The design keeps the runner behind one interface (`run_script(code, env) -> ScriptResult`) so swapping in a container changes one module.

**Self-correction.** A script that raises is sent back to Opus once with the traceback; a second failure returns the partial `report()` output and the error in the trace. Deep-tier calls record the script in the log so successful ones can be promoted to recipes.

## Logging, feedback and evaluation

Every invocation writes one record to `~/.oracle/log.sqlite`; the record is the raw material for every later improvement.

| Table | Columns |
| --- | --- |
| `calls` | call\_id, session, ts, query, intent, level, breadth, ranked\_tools (scores), plan (json), replans, escalated, wall\_ms, tokens\_in, tokens\_out, model\_calls, confidence, gaps |
| `ops` | call\_id, step\_id, op, params, ok, error, n\_results, ms, truncated, bytes, hop, parent (term or pointer that led here; crawl only) |
| `pointers` | session, pointer, call\_id, expanded\_at |
| `feedback` | call\_id, source (agent or human), useful, note, ts |

**Implicit signals**, derived nightly by `oracle report`:

- Re-query within 2 minutes in the same session with a reworded question: the first answer missed.
- `expand` on a pointer: the finding was used.
- Hit the step cap, replanned twice, or escalated: ranking or recipe gap.
- Ranked tool not in top 3 but used after replan: ranker miss; the query goes into `ranker_eval.yaml`.
- Deep-tier script succeeded with a shape seen three or more times: recipe candidate.

**Explicit signals.** `oracle feedback <call_id> --useful no --note "..."` is cheap enough that the main agent's instructions tell it to call it whenever a result was wrong or missing. A thumbs-down from the user on a conversation is joined to every call in that session.

**Golden set.** `golden.yaml` holds query → expected tools → expected pointers (at least one that must appear). `oracle eval` runs it, reports pass rate, ranker hit rate, mean tokens and wall time, and diffs against the last run. It runs on every manifest, recipe or prompt change. Failed queries from the logs are promoted into it, so the set grows from real misses.

Each `expect` clause is checked against what actually ran (`oracle/evals.py`): the primitive ops including recipe stages, the recipes used, the breadth flag, the resolved time window, and Coverage computed from execution. An unknown clause fails, so a typo cannot pass. Then a Sonnet judge reads the rendered answer against the `judge` rubric and fails only on a clear miss. `~/.oracle/eval/last.json` keeps each case's result and rendered answer, so a failure can be read without rerunning it. When a case fails, the fix goes into the oracle (ranker text, recipe `when` lines, prompts, fixtures), never into `golden.yaml`.

**Failure labelling.** Once a week, Sonnet labels low-signal calls with one of: wrong tool, bad params, right data but bad summary, needed more steps, question ambiguous. The distribution says what to fix next. This replaces guessing from individual anecdotes.

## Golden set: seed queries as acceptance tests

Milestone 2 is done when all four pass at `standard` with Coverage reporting what was and was not searched. Expected pointers are filled in once the registry is live; the tool list is the acceptance criterion until then. Two cases require `breadth: true` and grouping by source, so the plan-level breadth flag and grouped rendering were pulled forward from M3 into M2; the `--breadth` flag and the ranker's threshold mode stay in M3. At home the gate is `make gate-m2` (fixtures, unit tests, then `oracle eval` against the home profile; it spends model calls). First green run 2026-10-24: 4/4, ranker hit 4/4, about 17 s and 19.5K sub-agent tokens per case, 440–770 tokens per answer.

| Query | Shape | Expected tools (breadth) | Passes when |
| --- | --- | --- | --- |
| "important events from today" | breadth, time-bounded, needs ranking by importance | workiq.calendar, workiq.teams, acli.jira (updated today), argocd.app.history, kubectl.events | Findings grouped by source, ordered by the extractor's importance score, each with a pointer; Coverage names any source that was not configured. "Today" resolved in the user's timezone. |
| "how salesforce talks to cloud" | chain, two named endpoints | code-graph.find\_symbol ×2 → path\_between; code-search.search; acli.confluence.search | At least one call path with file:line hops, plus the design doc if Confluence has one. Replan expected when the first symbol search is ambiguous. |
| "where ownership transfer happens in code" | search → expand | recipe code.find\_and\_expand (code-search → code-graph.expand) | ≥ 1 code pointer whose expansion shows callers; confidence high when search and graph agree on the same symbols. |
| "who is andrew neville" | breadth, entity | recipe people.profile (workiq.people, acli.jira assignee, workiq.teams from:, git authors via code-graph if indexed) | Role and team from workiq, recent tickets and messages with pointers, and an explicit note when a source has no match (not a guess). |

Add to the set as misses appear. A query is only added with the tool list it *should* have used, so the set encodes intent, not current behaviour.

**Planned cases, added only once the phases that support them exist** (decided 2026-10-24):

- **Name follow-up, full name** (after M2's completeness check): "who wrote the ownership transfer design and what team are they on" → `acli.confluence.search` → author → `workiq.people.lookup`. Passes when the findings carry the author's role and team with pointers.
- **Name follow-up, first name only** (after M2): a fixture page that says only "Jon reviewed the gateway runbook", with two Jons in the directory. Passes when the findings list both candidates with the evidence for each and Coverage marks the ambiguity, or narrow to one using the page's evidence; never a silent pick.
- **Deep crawl, three hops** (after M4): see "Deep tier: crawl".

## Module layout and milestones

```
oracle/
  cli.py            # typer entrypoint: oracle, expand, feedback, tool, eval, report
  config.py         # ~/.oracle/config.toml + env overrides, model strings
  registry/
    schema.py       # pydantic models for manifests, ops, returns, recipes
    loader.py       # reads registry/tools/*.yaml, validates, caches
    importers/      # openapi.py, mcp.py, graphql.py, cli_draft.py (Sonnet-assisted)
    codegen.py      # emits oracle_tools/<tool>.py from manifests
  rank/
    index.py        # op documents, BM25, optional embeddings, RRF
    eval.py         # ranker_eval.yaml runner
  plan/
    models.py       # Plan, Step, bindings grammar
    planner.py      # PydanticAI agent: question + ranked ops -> Plan
    check.py        # completeness check agent
    distill.py      # Findings agent (Haiku)
    digest.py       # results for a model: summary fields + pointers inside a <data> block
  exec/
    executor.py     # one op: validation, auth, semaphores, paging, caps, redaction, pointers
    dag.py          # plan runner: dependency waves, recipe expansion and fallbacks, reuse
    bindings.py     # $s1.ids / $s1.pointers / $s1.results[0].x, recipe templates
    redact.py       # regex secret redaction on every op result
    runners/        # argv.py, http.py, mcp.py, db.py
    pointers.py     # pointer format, spill files, expand
  timewin.py        # "today", "this week", "past 3 days" -> absolute window in the user's zone
  evals.py          # oracle eval: golden clauses against what ran, plus the judge
  deep/
    script_agent.py # Opus agent that writes against oracle_tools
    runner.py       # run_script(): subprocess, env allowlist, AST check, limits
  log/
    store.py        # log.sqlite schema and writers
    report.py       # implicit signals, failure labelling
  workflows/
    protocol.py     # WorkflowContext, Budget, Trace, OracleClient, ToolsClient
    runner.py       # lazy import of entry points, recursion guard, budget sub-allocation
    ingest.py       # the tool ingestion workflow (probe, draft, verify, boundaries, evals, review)
  skills/
    loader.py       # role-static, tool-scoped, query-ranked skill selection
  prompts/          # planner.md, check.md, distill.md, script.md
  llm/
    claude_code.py  # claude-code:<alias> PydanticAI model over claude-agent-sdk (the subscription)
    models.py       # model string -> PydanticAI model
  ask.py            # one invocation: rank -> plan -> execute -> log -> Answer
  render.py         # output contract: Findings, Coverage, Refine, Trace (md | json)
  exec/selftest.py  # oracle tool test: run manifest examples, check expectations
adapters/bin/       # thin JSON CLIs over local tools (cmm-graph over codebase-memory-mcp)
profiles/           # repo-shipped environment profiles (home.toml, home/tools/*.yaml); ~/.oracle/profiles wins
registry/tools/*.yaml  # the work registry, filled by `oracle tool add`
registry/recipes/*.yaml
golden.yaml  ranker_eval.yaml
tests/            # unit tests per module + golden runner
  fakes/            # fake acli and workiq CLIs, their fixtures, and the acme fixture code repo
```

`uv` for packaging, `typer` for the CLI, `pydantic-ai` (as `pydantic-ai-slim[openai,anthropic]`) for the three agents, `rank_bm25` and optional `sentence-transformers` for ranking, `aiosqlite` for caches and logs, `pyyaml` for manifests, and `claude-agent-sdk` as the optional `claude-code` extra for the home model route. JSONPath is a hand-written subset (`$`, `.key`, `['key']`, `[n]`, `[*]`) in `registry/jsonpath.py`, so manifests are checked at load time. One process, asyncio throughout. Dev-only dependencies (a \`dev\` dependency group in \`pyproject.toml\`, never shipped): \`pytest\`, \`pytest-asyncio\`, \`ruff\`, \`pyright\`, \`radon\` or \`xenon\`, \`vulture\`, driven by a Makefile whose \`make lint\` runs lint, type, complexity, maintainability and dead-code checks and \`make test\` runs the suite and the golden set.

&#91;embedded content: milestones · 4 phases, 4 gates\]

Milestones run in order; each gate is a command (`oracle tool test`, `oracle eval`) that must be green before the next starts. M1 ships with four manifests (code-search, code-graph, acli, workiq) because the four seed queries need those. M2 is the first point at which Claude Code should use the oracle day to day.

The same plan as a table, for exports that drop the drawing:

| Milestone | Scope | Gate (command that must be green) |
| --- | --- | --- |
| M1 Core | Registry schema and loader; ranker (exact name + BM25); executor with argv runner, pagination, caps, pointers; manifests for code-search, code-graph, acli, workiq; `quick` tier; `oracle expand`; log schema | `oracle tool test` passes every manifest example; unit tests green |
| M2 Standard | Planner (Plan DAG), recipes (the four seeded), completeness check, distill, Coverage/Refine/Trace output; `standard` tier; feedback command; plan-level breadth and grouping (pulled forward from M3) | `oracle eval` passes all four golden cases at `standard` |
| M3 Scale | Remaining manifests (kubectl, argocd, es, mongo, custom scripts); breadth mode; `oracle report` implicit signals; ingestion workflow (`oracle tool add`); skills loader; MCP wrapper; environment profiles | `oracle tool test --ranker` top-3 hit rate at or above 90% on `ranker_eval.yaml`; golden set still green |
| M4 Deep | Codegen of `oracle_tools`; `oracle.sdk` (extract, fan_out, summarize, crawl, budget); the crawl with `--depth` and `--budget`; script agent and runner with v1 isolation; `deep` tier; workflow protocol and recursion guard; dreaming workflow after a few weeks of logs; container only if deep calls are frequent | One deep-tier script or crawl chain promoted to a recipe through the logs; the crawl acceptance case passes (if adopted); golden set still green |

### Developing on the personal laptop

The oracle is built on a personal laptop that lacks most work tools, then pulled onto the work laptop. Two pieces make the gates meaningful at home:

- **Fakes.** `tests/fakes/bin/acli` and `workiq` are stdlib Python CLIs with realistic `--help` and JSON output over fixtures in `tests/fakes/data/`, with times stored as offsets from now so "today" stays today. Their flags and output shapes are guesses at the real tools. They double as practice targets for `oracle tool add`. The fake Jira and Confluence understand a strict JQL/CQL subset: AND, OR, parentheses, `=`, `!=`, `~`, `!~`, `IN`, `NOT IN`, `resolution = Unresolved`, and time comparisons on `updated` (`-1d`, `-2w`, `startOfDay()`, `startOfWeek()`, ISO). A clause they cannot evaluate (`reporter was ...`) makes the fake exit 2 with the supported list, so a plan the fake cannot check is an op error, never a false pass.
- **Fixture code repo.** `tests/fakes/acme/` is a small codebase whose names match the fixtures (`transfer_ownership` and its callers for PLAT-412; `sync_salesforce_changes` → gateway client → ingestion gateway → event bus for INT-77). `make fixtures` indexes it into a repo-local vex db and into codebase-memory (Ollama is needed for the embeddings; `make test` does not need either).

`profiles/home.toml` sets `CMM_REPOS = "acme"` so the code graph sees only the fixture repo and evals do not depend on what else is indexed. It points at its own registry, `profiles/home/tools/`, with code-search (vex), code-graph (`adapters/bin/cmm-graph`) and the two fakes. `registry/tools/` stays the work registry and starts empty: on the work laptop each real tool is added with `oracle tool add`, one at a time, and the home manifests are never touched. Registry precedence is `ORACLE_REGISTRY`, then the profile's `registry`, then `config.toml`, then `registry/tools/`. Op names referenced by `golden.yaml` must survive the re-import: a unit test checks them against the home registry, and `oracle tool test` notes any golden op the active registry lacks.

**Registry problems and notes.** `oracle tool test` fails on problems: a manifest that does not parse or validate (including a JSONPath outside the supported subset), a name that does not match its file, an unreadable file or a missing registry directory. Expected states are notes and do not fail the gate: a refused mutating tool or op, a `chains_to` or `enum_from` reference to a tool that is not added yet, and golden ops the active registry lacks. A reference to an inactive tool is fine and silent; a reference to a missing op on a tool that is loaded is a typo and a problem. Naming a tool that is not active, or testing nothing at all, fails.

**Quick tier in M1.** The ranker shortlists tools, and one planner call (Sonnet) picks a single op and fills its params; there is no completeness check and no distill, so findings are each record's summary fields, minus values the pointer already carries. Levels that are not built yet run as quick and say so in the trace.

## Tool ingestion workflow

`oracle tool add <name | path | url> [--yes]` turns a new tool into an active manifest, a set of recipes and an eval set without hand-authoring. The governing rule: **every fact in the manifest is grounded in a real execution, not in documentation.** Drafting from `--help` alone produces invented flags and untested output shapes; running the tool is the only check that counts.

| Stage | What runs | Output | Grounding |
| --- | --- | --- | --- |
| 1 Probe | Detect type (CLI, OpenAPI URL, MCP server, GraphQL, script). Collect docs: `--help` recursively per subcommand, man page, README, spec. Record binary path and version. | `probe.json` | Real `--help` output, not training memory. |
| 2 Draft | Sonnet with the `manifest-authoring` skill drafts ops, params, argv templates, `when` / `not_when`, and one example per op. | `draft.yaml` | — |
| 3 Verify | Executor runs every example against the real tool. Ops whose example fails are repaired once by Sonnet with the error, then dropped. `returns` schema and `id_field` are inferred from the actual JSON, not declared. | `verified.yaml` | Every surviving op has run at least once. |
| 4 Boundaries | Compare the new tool's ops to the registry. For each overlap, Sonnet rewrites `when` / `not_when` on both sides. `oracle tool test --ranker` runs; any query the new tool now steals from an existing tool is flagged. | boundary diff | Ranker eval on the real registry. |
| 5 Evals | Generate 5–10 queries per op from real outputs: run a search, take a real id, build the expand case from it. Expected pointers come from the runs. Append to `ranker_eval.yaml` and `golden.yaml` as candidates. | eval candidates | Expected values observed, not predicted. |
| 6 Review | Show the manifest, boundary diff, and eval candidates. Activate on approval; `--yes` activates when verify and ranker eval are both green. | active manifest | Human gate on the text that drives selection. |

Stage 6 is the one stage not to automate away. `when` text is what decides tool selection, and the ingest agent cannot know the user's mental model of when to reach for ES rather than Mongo. The default is to require a glance; `--yes` exists for tools with no overlap.

**Failure handling.** No `--json` and no parseable table: the op gets `returns.format: text` and a note; the user can add a parser later. Interactive or prompting tool: refuse with a suggested wrapper script skeleton. Auth missing: the manifest is written with `auth.env` filled in and the tool is left inactive until the variable exists.

**Cost.** One ingestion is a workflow of roughly 10–30 model calls plus as many tool runs as there are examples. Minutes, not seconds, which is fine for something that happens once per tool. It is itself registered as a workflow (next sections), so it uses the same budget and logging as everything else.

## Skills for the oracle's agents

A skill is a markdown file in `skills/<name>/SKILL.md` with frontmatter saying which agent roles it applies to and when. The oracle's agents run in a fresh context on every call, so a skill costs its tokens once per call and is never buried; that is the property that makes skills cheap here and expensive in the main agent.

```yaml
---
name: jql
applies_to: [planner]
scope: tool:acli          # loaded whenever acli ops are in the shortlist
when: "Writing JQL for jira.search"
---
JQL cheat sheet: fields, operators, relative dates (`updated >= -1d`), ...
```

Three ways a skill reaches an agent:

| Mode | Example | Loaded when |
| --- | --- | --- |
| Role-static | `findings-format` for distill; `plan-format` for the planner; `manifest-authoring` for ingest | Always, for that role. |
| Tool-scoped | `jql`, `cql`, `es-query-dsl`, `cypher-readonly`, `kubectl-selectors` | Whenever that tool's ops are in the ranked shortlist. Attached in the manifest as `skills: [jql]`. |
| Query-ranked | `incident-triage`, `people-lookup` | Ranked alongside ops by the same ranker; the top 1–2 above a threshold are loaded. |

Tool-scoped skills are the high-value kind. Wrong or borrowed parameters are a documented failure mode of tool calling, and a query-language cheat sheet in front of the planner at the moment it writes JQL or an ES query is the direct fix. Ingestion drafts a tool-scoped skill in stage 2 when the tool has a query language.

**Presentation skills stay with the main agent.** A diagramming skill or a report-formatting skill belongs to whoever renders for the user, and the oracle does not. Its output contract is fixed (Findings, Coverage, Refine, Trace) so that the main agent can rely on it; a skill that changed the shape per call would break that. The one presentation skill the oracle owns is `findings-format`, which defines that contract.

**Budget.** Each loaded skill is capped at \~800 tokens; the planner sees at most four (one static, up to two tool-scoped, one query-ranked). `oracle eval` runs with and without each new skill before it is kept, so a skill that does not change results is removed rather than carried.

## Workflows as tools, and the recursion guard

A workflow is a Python entry point registered in the same manifest format as any tool, so the ranker, planner, executor, logging and budget treat it like an op. Ingestion, an incident investigation that gathers deploys, logs and tickets, or a deep code trace are all workflows.

```yaml
name: incident
kind: workflow
when: "Something broke in production and the user wants the timeline and likely cause."
ops:
  - name: investigate
    entry: oracle_workflows.incident:investigate
    params: {service: {type: str, required: true}, since: {type: str, default: "24h"}}
    returns: {format: findings}
    budget: {model_calls: 20, wall_s: 180}
```

**Protocol, not framework.** The oracle defines one small interface and depends on nothing else:

```python
async def investigate(params: InvestigateParams, ctx: WorkflowContext) -> Findings: ...

class WorkflowContext:
    oracle: OracleClient      # ctx.oracle.ask(query, level="standard") — depth-aware
    tools: ToolsClient        # ctx.tools.run("kubectl.events", ...) — direct op calls
    budget: Budget            # remaining model calls, tokens, wall time; shared across the tree
    log: Logger               # writes under the same trace_id
    trace: Trace              # trace_id, depth, chain of callers
```

A workflow body can be a plain async function, a `pydantic-graph` graph, or a rufalo flow behind a ten-line adapter that maps this context onto rufalo's. The core does not import rufalo; workflow modules are lazy-imported only when that workflow runs, because the oracle is a CLI invoked dozens of times per session and import time is paid on every invocation. The recommendation is to start with plain async functions, use rufalo where a workflow already exists in it, and reach for a graph library only when a workflow needs branching state that functions make ugly. This follows the pattern Anthropic and Cognition both arrived at: compose simple pieces, adopt a framework when a concrete workflow demands it.

**Dispatch.** Workflows have `when` text and are ranked with ops. The planner may choose one as a step, and the main agent can call one directly with `oracle run incident --service billing`. Workflows that call `ctx.oracle.ask()` get a nested oracle invocation that inherits the trace.

**Recursion guard.** The chain main agent → oracle → workflow → oracle → workflow is legal and useful, and it is the one place the design can loop forever. The guard is a `Trace` carried in-process and, for subprocesses, in the `ORACLE_TRACE` environment variable:

| Control | Rule | On breach |
| --- | --- | --- |
| Depth | `depth` increments on every nested oracle or workflow call; max 3 (configurable). | Refuse with the chain in the trace. |
| Cycle | The chain holds `(kind, name, hash(normalised query))`; a repeat is a cycle. | Refuse; Coverage says "would repeat `incident.investigate` for the same input". |
| Budget | One `Budget` per trace\_id stored in `log.sqlite`; every model call and op decrements it atomically, so sibling subprocesses share it. Child budgets are sub-allocations of the parent's. | Return partial results with Coverage stating the cutoff; never a silent stop. |
| Tier | A nested call cannot run at a higher tier than its parent and defaults one lower. | Clamped, noted in trace. |
| Preference | Planner and script prompts say: call an op directly when one does the job; `ctx.oracle.ask()` is for questions that need ranking across tools. | — |

Everything in a tree logs under one `trace_id`, so `oracle report` shows the whole call tree and where the budget went. The depth limit of 3 is a default to tune: with the current registry, nothing legitimate needs more than main agent → oracle → workflow → oracle.

## Caching, prior findings and remaining corner cases

The same `log.sqlite` (WAL mode, since the main agent may run oracle calls in parallel) carries four caches. Each exists to cut a specific cost, and each has a validity rule that is not just a clock.

| Cache | Key | Valid while | Saves |
| --- | --- | --- | --- |
| Op results | `(op, normalised params)` | Per-tool `cache_ttl` in the manifest: kubectl and ES near 0, Jira and Teams minutes, Confluence hours, code results until the touched files' git hashes change. | Repeat and overlapping tool runs within a session; time. |
| Plans | `(normalised query, intent, hash of ranked op set)` | Until a manifest, recipe or prompt changes; 7 days max. | One planner call on repeat questions. |
| Pointers | `(session, pointer)` | Session lifetime, plus 24 h for `expand`. | Re-fetching for `expand`. |
| Reported findings | `(session, pointer)` | Session lifetime. | Main-context tokens: a finding already delivered in this session is referenced ("see call 12") rather than repeated. |

**Prior findings as a search source.** Past distilled findings become one more registered tool, `findings.search`, ranked like any other and never privileged. Three rules keep it from contaminating answers with its own stale summaries, which is the documented failure of agent memory systems:

1. A prior finding is returned as a **lead, with provenance** (date, trace\_id, the pointers it rested on), never as an answer. The completeness check treats it as something to re-verify through the pointers, unless its validity rule still holds.
2. Validity is keyed to the source, not only to time: a code finding carries the commit hash and file hashes it was derived from and is valid while those match; a people finding expires in a day; a ticket finding in an hour.
3. Findings are stored with their Coverage. A prior finding that said "not searched: jira" cannot satisfy a question that needs Jira.

If `findings.search` wins a query outright, the output says so and the Refine section offers the fresh run. Hit rate and the rate at which fresh runs contradict prior findings are logged; a high contradiction rate shortens TTLs.

**Remaining corner cases**, each with its default:

- **Prompt injection through tool output.** Confluence pages, Teams messages and tickets can contain text aimed at the model. Tool output is data: the planner on replan and the check see only summary fields; distill sees capped slices inside a delimited data block; prompts state that instructions inside results are to be reported, not followed. Default: on, not configurable.
- **Secrets in output.** Env-var auth keeps secrets out of argv, but `kubectl get secret` or a config dump can return them. Default: a per-tool denylist of resources and a regex redaction pass (tokens, keys, connection strings) on every op result before caching or distilling. The redaction pass ships in M2 (`exec/redact.py`); the per-tool denylist comes with the kubectl manifest in M3.
- **Duplicate entities across sources.** The same ticket via acli and via ES, the same person via workiq and git. Default: distill dedupes on pointer identity and on normalised ids (ticket key, email); grouped, not repeated.
- **Environment profiles.** Home has a subset of the tools. Default: `~/.oracle/profiles/<name>.toml` (falling back to the repo's `profiles/`) selects which manifests are active and adds `path_prepend` and `env` for tool subprocesses; inactive tools appear in Coverage as "not available here", not as gaps. Pulled forward into M1 in minimal form so the personal laptop can run the gates.
- **Tool drift.** A CLI upgrade changes a flag. Default: the manifest records the binary version; `oracle tool test` runs weekly as a scheduled task and on version change.
- **Relative time.** "Today", "since yesterday", "this sprint" resolve in the user's timezone from the config, and the resolved absolute range is printed in the trace. M2 resolves today, yesterday, this week, last week and "past/last N hours/days/weeks" deterministically before planning; "this sprint" needs the tracker and stays with the planner.
- **Binary or non-UTF-8 output.** Default: write to a temp file, return a pointer and the first 1 KB as hex; never hand bytes to a model.
- **Partial failures in breadth mode.** Default: never fail the call; every failed source is named in Coverage with its error class.

## Dreaming: the nightly improvement workflow

`oracle dream` is a scheduled workflow that reviews the day, proposes changes to the registry, skills, recipes and memory, evaluates every proposal against the golden set, and promotes only what passes. It is sequenced after M3: it needs a few weeks of logs and a green golden set before its proposals mean anything.

The governing rule: **dreaming proposes, evals and the user dispose.** Self-modifying agents drift when their changes are judged by the same model that made them. Reported results on evolving agent contexts (ACE, 2025) show gains when updates are incremental and checked, and two named failure modes when they are not: brevity bias (rewrites that compress away the detail that mattered) and context collapse (a monolithic rewrite that loses accumulated knowledge). So dreaming edits by delta, never by rewrite, and every delta is scored before it lands.

| Stage | Input | Output |
| --- | --- | --- |
| 1 Ingest | Day's `log.sqlite` rows; Claude Code session transcripts (local JSONL), parsed for oracle calls, the user's next message, and shell commands the main agent ran directly. | A day table: query, oracle result, what happened next. |
| 2 Label | Sonnet classifies each call: confirmed useful, corrected by user, re-queried, ignored, failed (wrong tool / bad params / bad summary / too few steps / ambiguous). The user's next message is the external signal; the model's own confidence is not. | Labelled calls. |
| 3 Propose | Typed proposals, each citing the calls that motivate it. | See classes below. |
| 4 Evaluate | Apply each proposal to a scratch copy of the registry; run `oracle eval` and the ranker eval; measure tokens and wall time. | Pass / regress per proposal. |
| 5 Promote or queue | Safe classes that pass are applied and committed. Everything else is written to `dreams/<date>.md` with diffs and evidence for a two-minute morning review. | Commits plus a review file. |
| 6 Maintain memory | Expire findings past TTL; re-verify code findings whose file hashes changed; merge duplicate entity profiles (people, services) keeping provenance; rebuild the ranker index and embeddings. | Smaller, truer store. |
| 7 Report | What changed, what is queued, and the week's trend of pass rate, replan rate, cache hit rate, contradiction rate. | One page. |

**Proposal classes and who decides**

| Class | Evidence that triggers it | Auto-promote? |
| --- | --- | --- |
| New eval case | A call labelled corrected or re-queried | Yes, as a candidate flagged for confirmation |
| Recipe from a repeated chain | Same op sequence in 3+ successful calls | Yes, if its generated tests pass |
| TTL adjustment | Contradiction rate between prior findings and fresh runs | Yes, within bounds (never longer than the manifest maximum) |
| `when` / `not_when` edit | Ranker misses on a tool | No: review |
| Skill delta | Repeated param errors for one tool | No: review; delta only, capped size |
| New tool candidate | A CLI the main agent ran directly via Bash 3+ times that is not registered | No: queues an `oracle tool add` suggestion |
| Prompt change | Systematic bad-summary labels | No: review, and only as a delta |
| Memory merge or expiry | Duplicates, failed re-verification | Yes |

**What dreaming never does.** It does not rewrite a prompt or skill wholesale, does not add a skill that `oracle eval` shows makes no difference, does not learn from its own outputs without a user signal, and does not touch anything mutating. Its budget is fixed (model calls and wall time) and it runs under its own trace\_id like any workflow.

**Transcripts.** Claude Code's local session files are the richest signal because they contain the user's reaction, which the oracle never sees at call time. They stay on the machine; dreaming reads them, extracts the day table, and discards the rest. At work this is the one place sensitive content enters an oracle process, so the parser keeps only query, result summary and the next user message.

**Measuring whether dreaming helps.** The week-over-week trend in the report is the test: golden pass rate up or flat, replan rate down, main-context tokens per call down, contradiction rate down. If a month of dreaming moves none of these, turn it off; the logs and manual `oracle report` remain.

## Assumptions and open questions

Assumptions Claude Code should treat as settled unless told otherwise:

- Single user, single machine, read-only tools. No multi-tenancy, no shared registry, no auth beyond environment variables.
- The graph DB and the semantic search CLI already produce JSON. If not, their manifests get a `table` or `text` parser in M1.
- Sessions are identified by an id Claude Code passes with `--session`; without one, pointers live for the process lifetime only.
- Bedrock models are reached through the LiteLLM proxy with OpenAI-compatible calls; direct Bedrock is a fallback, not the default.
- The MCP wrapper is optional and comes after the CLI works in Bash.
- `k9s` is out of scope; `kubectl` covers it.

Open questions, each with a default so work does not block:

- [ ] **Replan rate.** If more than \~30% of standard calls replan, the chains are mostly judgment and the step loop should replace plan-then-execute for those tools. Default: ship plan-then-execute, measure, decide at M3.
- [ ] **Embeddings in the ranker.** Not needed at \~30 ops. Default: BM25 only until `ranker_eval` top-3 drops below 90%.
- [x] **Home billing.** Resolved 2026-10-24: home uses the Claude subscription through Claude Code (`claude-code:<alias>` models), as rufalo does; work uses Bedrock through LiteLLM.
- [ ] **Pointer stability for Teams and mail.** workiq ids may not be stable across days. Default: pointer carries a date and the expand re-searches if the id is gone.
- [ ] **Importance ranking for "events from today".** Rules-based (mentions, assigned to me, deploy failures) or model-based. Default: the extractor ranks with a short rubric in `distill.md`; revisit with feedback data.
- [ ] **Deep-tier container.** Only if deep calls exceed a few per day. Default: subprocess isolation as specified.
- [ ] **Which `custom-*` scripts to register first.** Default: whichever the four seed queries need; the rest in M3.
