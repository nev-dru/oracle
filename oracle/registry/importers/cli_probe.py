"""Stage 1 of `oracle tool add` for a CLI: collect real `--help` output, recursively and bounded.

Subcommands are read from the help text itself (cobra/kubectl "... Commands:" sections, click
"Commands:", argparse `{a,b}`), never from model memory. A subcommand whose name is a mutating
verb is never run, not even with `--help`: some CLIs execute the subcommand anyway.
"""

import asyncio
import contextlib
import re
import shutil
from dataclasses import asdict, dataclass, field

from oracle.registry.safety import mutating_word

SKIP = frozenset({"help", "completion", "version", "docs", "plugin", "alpha"})
SECTION = re.compile(r"^\s*(?:[A-Za-z][\w ()/-]*\s)?(?:sub)?commands?(?: \([^)]*\))?:\s*$", re.I)
ENTRY = re.compile(r"^\s{1,8}([a-z][a-z0-9_-]*)(?:,\s*[a-z][a-z0-9_-]*)*(?:\s{2,}\S.*)?$")
BRACES = re.compile(r"\{([a-z][a-z0-9_,-]*)\}")


@dataclass
class Probe:
    tool: str
    binary: str
    path: str | None
    version: str | None = None
    helps: dict[str, str] = field(default_factory=dict)  # "" is the root; "jira workitem"
    refused: list[str] = field(default_factory=list)  # mutating subcommands, never run
    truncated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def subcommand_names(text: str) -> list[str]:
    names: list[str] = []
    for group in BRACES.findall(text):
        names += group.split(",")
    in_section = False
    for line in text.splitlines():
        if SECTION.match(line):
            in_section = True
            continue
        if in_section:
            m = ENTRY.match(line)
            if m:
                names.append(m[1])
            else:  # a blank line or any other text ends the command list
                in_section = False
    return list(dict.fromkeys(n for n in names if n not in SKIP))


async def _run(argv: list[str], env: dict[str, str], timeout: float) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except OSError:
        return ""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return ""
    return out.decode("utf-8", "replace")


async def probe_cli(
    tool: str,
    binary: str,
    env: dict[str, str],
    *,
    max_depth: int = 3,
    max_commands: int = 40,
    max_chars: int = 60_000,
    timeout: float = 10.0,
) -> Probe:
    probe = Probe(tool=tool, binary=binary, path=shutil.which(binary, path=env.get("PATH")))
    if probe.path is None:
        return probe
    probe.version = await _version(binary, env)
    queue: list[list[str]] = [[]]
    used = 0
    while queue:
        words = queue.pop(0)
        if len(probe.helps) >= max_commands or used >= max_chars:
            probe.truncated = True
            break
        text = (await _run([binary, *words, "--help"], env, timeout))[:8000]
        if not text.strip() or (words and text in probe.helps.values()):
            continue  # empty, or the same text again: the CLI ignored the subcommand
        probe.helps[" ".join(words)] = text
        used += len(text)
        if len(words) < max_depth:
            queue += _children(probe, words, text)
    return probe


def _children(probe: Probe, words: list[str], text: str) -> list[list[str]]:
    """Subcommands to probe next; mutating ones are recorded and never run."""
    out = []
    for name in subcommand_names(text):
        path = [*words, name]
        if mutating_word([name]):
            probe.refused.append(" ".join(path))
        elif " ".join(path) not in probe.helps:
            out.append(path)
    return out


async def _version(binary: str, env: dict[str, str]) -> str | None:
    text = (await _run([binary, "--version"], env, 5.0)).strip()
    first = text.splitlines()[0][:200] if text else ""
    return None if re.search(r"(?i)error|unknown|usage", first) else first or None
