---
name: incident-triage
applies_to: [planner, check]
scope: query
when: "Something broke, is down, erroring, slow or degraded; an outage, incident, 5xx or failed deploy"
---
Incident questions need the timeline from several sources at once (breadth):
- what changed: recent deploys and sync state (argocd);
- what the cluster says now: pod status and Warning events (kubectl);
- what failed: error logs and counts for the service (es);
- who is on it: open tickets and the incident runbook (acli);
- what people said: Teams messages since the start (workiq).
Ask each source about the same service and time window; the check should look for a deploy
shortly before the first errors.
