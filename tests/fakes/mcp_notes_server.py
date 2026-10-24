"""A tiny stdio MCP server for tests: two read-only tools and one that writes a marker file if it
is ever called, so a test can prove the importer never runs a mutating tool."""

import json
import os
import sys

from mcp.server.mcpserver import MCPServer

NOTES = [
    {"id": "n1", "title": "Gateway timeouts", "body": "Raise the batch buffer before the window"},
    {"id": "n2", "title": "Ownership transfer", "body": "Always write the audit row"},
]
server = MCPServer("notes")


@server.tool()
def search_notes(query: str) -> str:
    """Search notes by words in the title or body."""
    words = query.lower().split()
    hits = [n for n in NOTES if all(w in (n["title"] + " " + n["body"]).lower() for w in words)]
    return json.dumps({"notes": hits})


@server.tool()
def list_notes() -> str:
    """List every note."""
    return json.dumps({"notes": NOTES})


@server.tool()
def rebuild_index() -> str:
    """Rebuild the search index (needs no arguments and changes state)."""
    with open(os.environ.get("NOTES_MARKER", "/dev/null"), "w") as f:
        f.write("rebuilt")
    return "rebuilt"


@server.tool()
def delete_note(note_id: str) -> str:
    """Delete a note."""
    with open(os.environ.get("NOTES_MARKER", "/dev/null"), "w") as f:
        f.write("deleted")
    return "deleted"


if __name__ == "__main__":
    server.run("stdio")
    sys.exit(0)
