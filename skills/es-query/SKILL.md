---
name: es-query
applies_to: [planner, check]
scope: tool:es
when: "Writing an Elasticsearch query string for es.search or es.count"
---
Elasticsearch query strings for `es.search` / `es.count`:
- `field:value` matches exactly: `level:error`, `status:500`, `service:ingestion-gateway`,
  `request_id:req-abc123`. Bare words search the message.
- Combine with AND / OR: `level:error AND service:billing-api`.
- `--index` takes a name or pattern: `logs-*` for every service, `logs-gateway` for one.
- `since` is a relative duration: `15m`, `1h`, `1d`. For "today" use the hours since midnight.
- Use `es.count` for "how many"; `es.search` with a small `size` for examples.
