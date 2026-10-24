# Registry

`tools/` holds the active manifests when no profile names its own registry. It starts empty:
on the work laptop each real tool is added with `oracle tool add`, which writes here.

The personal laptop uses `profiles/home/tools/` instead (vex, codebase-memory and the fake
acli/workiq), selected by `ORACLE_PROFILE=home`.
