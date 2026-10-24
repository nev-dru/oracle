"""Cloud side: validates incoming events and writes them to the event bus."""

from cloud.event_bus import publish


def ingest_events(token: str, events: list[dict]) -> int:
    if not token:
        return 0
    valid = [e for e in events if "id" in e]
    for event in valid:
        publish("salesforce.changes", event)
    return len(valid)
