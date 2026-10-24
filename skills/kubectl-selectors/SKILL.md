---
name: kubectl-selectors
applies_to: [planner, check]
scope: tool:kubectl
when: "Choosing kubectl resources, namespaces and selectors"
---
kubectl:
- Live state only: what is running, restarting or warning now. History and searches over past
  logs belong to es; deploy history to argocd.
- Pass the namespace when the question names one (`payments`, `prod`, `billing`); otherwise set
  `all_namespaces: true`.
- Labels select an app: `app=ingestion-gateway`.
- Crashes and restarts: pods (status, restarts) plus Warning events (`types: Warning`).
- Never ask for secrets.
