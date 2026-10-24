"""HTTP handlers for the accounts service."""

from accounts.ownership import batch_transfer, transfer_ownership


def handle_transfer_request(body: dict) -> dict:
    """POST /resources/{id}/owner"""
    transfer_ownership(body["resource_id"], body["new_owner"], body["actor"])
    return {"status": "ok"}


def handle_bulk_transfer(body: dict) -> dict:
    """POST /resources/owner:batch"""
    return {"moved": batch_transfer(body["resource_ids"], body["new_owner"], body["actor"])}


def handle_offboarding(user_id: str, successor: str, owned: list[str]) -> None:
    """Called when an employee leaves: every resource they own moves to the successor."""
    for rid in owned:
        transfer_ownership(rid, successor, "system:offboarding")
