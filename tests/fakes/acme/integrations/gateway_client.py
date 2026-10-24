"""HTTP client for the cloud ingestion gateway."""

from cloud.ingestion_gateway import ingest_events


def post_batch(events: list[dict], retries: int = 3) -> int:
    for _attempt in range(retries):
        accepted = ingest_events(refresh_token(), events)
        if accepted:
            return accepted
    return 0


def refresh_token() -> str:
    return "service-token"
