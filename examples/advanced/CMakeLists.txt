# Advanced examples — only subdirectories that have a CMakeLists.txt are built.
# Markdown/Python-only examples are self-contained and require no build step.
#
# `power_retention/` (the only CMake subproject here) was retired to
# `archive/examples/advanced/power_retention/` on 2026-09-17: its op is
# promoted into the canonical dialect (`tessera.power_attn` / `tessera.retention`,
# LA-4), and its CUDA kernel was a scaffold that never compiled. Nothing under
# this directory is built any more.
