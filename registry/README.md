# Registry

The work registry lives outside the repo, in `~/.oracle/registry/tools/` (the default when no
profile or config names another), because its manifests describe internal tools: nothing from
it is committed here. It starts empty; on the work laptop each real tool is added with
`oracle tool add`, which writes there. Work recipes go beside it, in `~/.oracle/registry/recipes/`.

The personal laptop uses `profiles/home/tools/` instead (vex, codebase-memory and the fake
acli/workiq), selected by `ORACLE_PROFILE=home`.

`recipes/README.md` lists the four seeded chains to rebuild once their tools are added, and
what to check in each.
