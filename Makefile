.PHONY: build lint test fixtures gate-m1 gate-m2 gate-m3 gate-m4 accept

PY := oracle tests
FIXTURE_REPO := $(CURDIR)/tests/fakes/acme
export VEX_DB := $(CURDIR)/.fixtures/vex.db

build:
	uv sync --all-extras
	uv build -q

lint:
	uv run --all-extras ruff check $(PY)
	uv run --all-extras ruff format --check $(PY)
	uv run --all-extras pyright
	uv run --all-extras xenon --max-absolute B --max-modules B --max-average A oracle
	uv run --all-extras radon mi -s --min B oracle | (! grep -E " - [CDEF] ")
	uv run --all-extras vulture oracle vulture_allowlist.py --min-confidence 80

test:
	uv run --all-extras pytest -q

# Index the fixture repo into a repo-local vex db and into codebase-memory (needs Ollama for embeddings).
fixtures:
	mkdir -p .fixtures
	vex index --dir $(FIXTURE_REPO)
	vex refresh
	codebase-memory-mcp cli index_repository '{"repo_path": "$(FIXTURE_REPO)"}' 2>/dev/null >/dev/null

# M1 gate: every manifest example passes against the home profile, and unit tests are green.
gate-m1: fixtures test
	ORACLE_PROFILE=home uv run oracle tool test

# M2 gate: all golden cases pass at their level against the home profile. Spends model calls:
# set ORACLE_MODEL_PLANNER and ORACLE_MODEL_EXTRACT (or [models] in ~/.oracle/config.toml) first.
gate-m2: fixtures test
	ORACLE_PROFILE=home uv run --all-extras oracle eval

# M3 gate: ranker top-3 >= 90% on ranker_eval.yaml, every manifest example green, golden still
# green. The golden part spends model calls (see gate-m2).
gate-m3: fixtures test
	ORACLE_PROFILE=home uv run --all-extras oracle tool test
	ORACLE_PROFILE=home uv run --all-extras oracle tool test --ranker
	ORACLE_PROFILE=home uv run --all-extras oracle eval

# M4 gate: a crawl chain promoted to a recipe through the logs (`oracle dream`, then
# `oracle dream --accept`), and everything M3 checked, including the crawl acceptance case.
# The golden part spends model calls (see gate-m2).
gate-m4: fixtures test
	# a recipe `oracle dream --accept` installed: only dream writes "Learned from N crawls"
	grep -l "Learned from [0-9]* crawls" profiles/home/recipes/learned.*.yaml
	ORACLE_PROFILE=home uv run --all-extras oracle tool test
	ORACLE_PROFILE=home uv run --all-extras oracle tool test --ranker
	ORACLE_PROFILE=home uv run --all-extras oracle eval

# Acceptance on the active registry, for the work laptop: no ORACLE_PROFILE, so registry/tools/
# and golden.yaml's `profiles.default` blocks. The gates above run the home profile only. Spends
# model calls (see gate-m2).
accept: test
	uv run --all-extras oracle tool test
	uv run --all-extras oracle tool test --ranker
	uv run --all-extras oracle eval
