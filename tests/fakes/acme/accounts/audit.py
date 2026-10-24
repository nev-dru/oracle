def write_audit(event: str, resource_id: str, old: str, new: str, actor: str) -> dict:
    return {"event": event, "resource": resource_id, "old": old, "new": new, "actor": actor}
