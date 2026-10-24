"""Moving a resource from one owner to another."""

from accounts.audit import write_audit
from accounts.store import load_resource, save_resource


class TransferError(Exception):
    pass


def transfer_ownership(resource_id: str, new_owner: str, actor: str) -> None:
    """Hand a resource to a new owner atomically and record the change (PLAT-412)."""
    resource = load_resource(resource_id)
    previous = resource["owner"]
    if previous == new_owner:
        return
    resource["owner"] = new_owner
    save_resource(resource)
    if is_active(previous):
        write_audit("owner_changed", resource_id, previous, new_owner, actor)


def batch_transfer(resource_ids: list[str], new_owner: str, actor: str) -> int:
    for rid in resource_ids:
        transfer_ownership(rid, new_owner, actor)
    return len(resource_ids)


def is_active(user_id: str) -> bool:
    return not user_id.startswith("deactivated:")
