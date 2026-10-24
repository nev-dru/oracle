_DB: dict[str, dict] = {}


def load_resource(resource_id: str) -> dict:
    return _DB.setdefault(resource_id, {"id": resource_id, "owner": "nobody"})


def save_resource(resource: dict) -> None:
    _DB[resource["id"]] = resource
