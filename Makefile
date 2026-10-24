.PHONY: build lint test fixtures gate-m1 gate-m2

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
