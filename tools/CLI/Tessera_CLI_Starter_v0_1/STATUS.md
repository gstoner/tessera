# Status: `archived`

Tracked by `python/tessera/compiler/tools_manifest.py`.

`tools/CLI/Tessera_CLI_Starter_v0_1` is a historical standalone CLI
starter suite.  It is kept in-tree as a reference for its small CMake
layout, shared argument helpers, and seven mock `tessera-*` commands,
but it is no longer part of the active compiler toolchain.

The active tool surfaces are:

- `tools/tessera-opt`
- `tools/tessera-translate`
- `tools/profiler`
- `tools/roofline_tools`

Do not add new compiler support claims to this archived starter.  If a
behavior is needed by current Tessera users, implement and test it in
the active tool surface instead.
