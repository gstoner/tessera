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

## Registry and lifecycle drift gates

Treat every new dtype, operation, diagnostic code, target, pass, and audit plan
state as a cross-registry change, not a local implementation detail.

- Diagnostics: every stable all-caps `CODE_NAME:` emitted from C++ or Python
  must be registered in `python/tessera/compiler/diagnostic_codes.py` with its
  origin, severity, summary, fix hint, specification link, implementation
  language, and status. Add or update pass metadata when the code belongs to a
  registered pass. Never merge an emission site without running
  `tests/unit/test_diagnostic_code_registry.py` and
  `tests/unit/test_pass_metadata.py`.
- Dtypes: register canonical spelling and aliases in `python/tessera/dtype.py`,
  then update every applicable target capability, architecture selector,
  physical storage/packing contract, runtime ABI, generated target view, and
  numerical oracle. A target-supported or planned-gated dtype is not
  automatically a first-class Graph IR storage dtype. Run
  `tests/unit/test_canonical_dtype.py` and
  `tests/unit/test_tensor_attributes_dtype_audit.py`, plus the affected
  backend's dtype/target totality tests.
- Operations: register public operations through the canonical op catalog and
  runtime registry, then update the applicable Graph IR mapping, ODS dialect
  declaration/registration, effects, dtype/layout contract, numeric policy,
  autodiff rules, backend manifest/capability rows, execution/conformance
  state, test-coverage registry, and generated documentation. Use explicit
  `not_applicable`, `planned`, or `artifact_only` states where execution does
  not exist; never imply support from parser or artifact evidence. Run
  `tests/unit/test_operator_registry_foundation.py`,
  `tests/unit/test_tensor_attributes_dtype_audit.py`, and the affected dialect,
  backend, autodiff, conformance, and generated-dashboard drift tests.
- Targets and passes: update all totality surfaces together—capabilities,
  selector tables, pipeline registry, execution matrix, CLI target lists,
  generated dashboards, and tests. Regenerate checked-in derived files with
  their owning generator instead of hand-editing them.
- Audit plans: authored documents with `audit_role: plan` may use only
  `plan_state: open`, `landing`, or `closed`. Use `landing` while implementation
  or referenced follow-on work is still active. A `closed` plan must be moved
  out of the live audit tree into the theme's `archive/` and summarized by the
  owning audit. Run `tests/unit/test_audit_docs.py` after any audit frontmatter
  or lifecycle edit.

Before publishing a PR that touches any of these surfaces, run the applicable
focused drift gates in host WSL before the broader unit/validation lanes. Treat
`validate-required` as a downstream fan-in: diagnose and fix the failing unit,
audit, lint, or build lane rather than changing the fan-in gate or weakening a
test state.

## Cross-backend work coordination

The active architecture queues are:

- `docs/audit/backend/apple/todo.md`
- `docs/audit/backend/nvidia/todo.md`
- `docs/audit/backend/rocm/todo.md`

Before starting backend compiler or runtime work, read all three plans and
identify the owning work-item ID.

When a change affects shared IR, ABI, dtype/op registration, diagnostics,
numerical policy, test infrastructure, benchmark schemas, or runtime contracts,
the same PR must assess all three backends and update each affected plan. Record
the sibling-backend outcome as follow-up required, parity validated, or not
applicable with an architecture-specific reason.

Use PRs as synchronization points. Every cross-backend PR must name its owning
item, shared contracts changed, backend-plan updates, validation performed, and
missing exact-device evidence. Link follow-up hardware PRs with a common
cross-backend synchronization key.

Never transfer physical schedules between architectures or mark sibling work
complete without exact-device evidence from that backend's required host. A
shared-contract PR may land with host-free tests and plan updates before its
linked exact-device follow-ups; do not hold unrelated backend work in one giant
multi-hardware PR. After a coordinating PR merges, reread all three plans before
selecting the next architecture action.

## RDNA ISA data archive

Use `docs/reference/isa/rdna/` as the does-this-op-exist-on-my-target truth before emitting. It is a structured, regenerable extraction of AMD's RDNA3 / RDNA3.5 / RDNA4 ISA guides plus the Micro Engine Scheduler. Each version has an instruction database at `<ver>/instructions.json` (opcodes and pseudocode) and microcode encoding bit-fields at `encodings.json`; the cross-version opcode matrix is at `cross_version/instruction_matrix.{json,md}`.

`gfx1151` is RDNA3.5: it supports WMMA F16/BF16/IU8/IU4, but does not support FP8/BF8 WMMA. FP8/BF8 WMMA and sparse SWMMAC are RDNA4-only.

JSON is machine truth; Markdown is a mirror. Regenerate the archive with `tools/build_archive.py` (no network). The MES scheduler write-up is at `mes/SCHEDULER_OVERVIEW.md`.
