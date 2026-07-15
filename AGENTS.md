## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Test execution environment

All tests for this project must run in the host WSL environment, never in the Codex sandbox. Use an escalated/unsandboxed command for every test invocation, including targeted tests, smoke tests, and full suites. Do not treat or report a sandbox test run as project validation. For CUDA and ROCm tests, verify that the intended host GPU and toolchain are visible from WSL before running; if they are not, report the host-environment blocker instead of substituting a sandbox run.

## GitHub execution environment

All Git and GitHub publishing operations for this project must run in the host WSL environment, never in the Codex sandbox. This includes checking `gh` authentication, creating or switching PR branches, staging and committing changes, pushing branches, inspecting remote PR state, and creating or updating pull requests. Use escalated/unsandboxed commands for these operations so they use the host WSL Git configuration, credentials, and GitHub CLI session.

## RDNA ISA data archive

Use `docs/reference/isa/rdna/` as the does-this-op-exist-on-my-target truth before emitting. It is a structured, regenerable extraction of AMD's RDNA3 / RDNA3.5 / RDNA4 ISA guides plus the Micro Engine Scheduler. Each version has an instruction database at `<ver>/instructions.json` (opcodes and pseudocode) and microcode encoding bit-fields at `encodings.json`; the cross-version opcode matrix is at `cross_version/instruction_matrix.{json,md}`.

`gfx1151` is RDNA3.5: it supports WMMA F16/BF16/IU8/IU4, but does not support FP8/BF8 WMMA. FP8/BF8 WMMA and sparse SWMMAC are RDNA4-only.

JSON is machine truth; Markdown is a mirror. Regenerate the archive with `tools/build_archive.py` (no network). The MES scheduler write-up is at `mes/SCHEDULER_OVERVIEW.md`.
