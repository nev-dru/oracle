# Registry

`tools/` holds the active manifests when no profile names its own registry. It starts empty:
on the work laptop each real tool is added with `oracle tool add`, which writes here.

The personal laptop uses `profiles/home/tools/` instead (vex, codebase-memory and the fake
acli/workiq), selected by `ORACLE_PROFILE=home`.

`recipes/` holds the work recipes; it starts empty too. `recipes/README.md` lists the four
seeded chains to rebuild once their tools are added, and what to check in each.
