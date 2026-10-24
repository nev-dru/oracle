"""Outbound Salesforce connector: pushes change events toward the cloud (INT-77)."""

from integrations.gateway_client import post_batch


def sync_salesforce_changes(changes: list[dict]) -> int:
    """Entry point the Salesforce outbound trigger calls on every batch window."""
    batch = [normalise_change(c) for c in changes]
    return post_batch(batch)


def normalise_change(change: dict) -> dict:
    return {"object": change["sobject"], "id": change["Id"], "fields": change.get("fields", {})}
