_TOPICS: dict[str, list[dict]] = {}


def publish(topic: str, event: dict) -> None:
    _TOPICS.setdefault(topic, []).append(event)
