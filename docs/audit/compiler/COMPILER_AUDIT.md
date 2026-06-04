# Compiler Audit

This document consolidates the compiler audit material that previously lived in
multiple root audit documents and compiler archive files.

## Finished

- **Canonical driver:** `canonical_compile` and `CompileResult` are the common
  contract for compilation results.
- **Runtime handoff:** `@jit` and `runtime.launch()` consume canonical compile
  metadata rather than inventing a second path.
- **Capability gates:** legality, codegen, toolchain, link, runtime ABI,
  hardware smoke, and numerical gates report named failure axes.
- **Conformance matrix:** op-target proof is rendered in
  `../op_target_conformance.md` and drift-gated.
- **Schedule to Tile metadata:** mesh, layout, placement, artifacts, and related
  metadata survive lowering.
- **C++ pass honesty:** `LowerScheduleToTargetPass` stopped pretending to be an
  implemented lowering pass.
- **Tile to Apple parity:** C++ Apple status tags match the Python/runtime Apple
  envelope.
- **Dynamic control flow:** unsupported dynamic control flow now gets explicit
  diagnostics and fallback behavior.
- **Frontend bugs:** AugAssign sub/div lowering, ROCm sub-arch gates, and
  Darwin arm64 platform checks were fixed.
- **Compiler correctness tests:** pass-order and oracle fixtures cover string
  parsing, MLIR pass order, halo execution, CorrDiff IR visibility, spectral,
  linear attention, and Apple runtime pipeline order.
- **CSV-canonical generated dashboards + sprint regen (2026-06-04).**
  `runtime_abi` and `verifier_coverage` now emit a machine-readable CSV
  (`docs/audit/generated/*.csv`, stable-sorted, byte-diffable) as the
  drift-gated artifact, with the human Markdown demoted to a non-byte-gated
  companion. Both are wired into `scripts/check_generated_docs.sh`, which gained
  a `--write` mode so `scripts/check_generated_docs.sh --write` regenerates
  every registered doc at sprint end. This retired the byte-exact-markdown
  drift gates that had been reddening CI (`runtime_abi.md` was stale 234 vs 241
  symbols). The four Apple CPU+GPU state docs were also consolidated into the
  single reference `docs/apple_backend.md`.

## Still Open

- **Program identity — component-op vectors + gating landed (2026-06-02);
  richer metadata still open.** `CompileResult` now carries ``component_ops``
  (the whole-program distinct op vocabulary), ``program_executable`` (gated
  component-by-component, not just the primary op), and ``component_blockers``
  ((op, failing-gate) pairs), surfaced in ``to_dict`` / ``to_runtime_artifact``.
  Locked by `tests/unit/test_canonical_component_ops.py`. Still open: graph
  outputs, ``effects``, ``shape_envelope``, and ``fusion_groups`` in the
  canonical metadata.
- **Fusion intent is too late.** Target IR and runtime still rediscover patterns
  that should be represented by Schedule/Tile/Target metadata.
- **Layout and binding contracts are uneven.** Graph/Schedule/Tile/Target IR
  need stronger dtype, layout, aliasing, and buffer-binding contracts.
- **Complete claims need fixtures.** A completed backend claim should resolve to
  an explicit compare fixture, `hardware_verified` row, or packaged validation.
- **Compiler specs can still drift.** Generated dashboards must remain the
  source of counts; prose docs should link, not duplicate snapshots.
- **Generated-doc regeneration + drift gating is fragmented (2026-06-04).**
  There are ~24 generated artifacts under `docs/audit/generated/` (+ the root
  `op_target_conformance.md` / `standalone_primitive_coverage.md`), but drift is
  enforced across *two parallel entry points* — `scripts/check_generated_docs.sh`
  (7 docs, pre-commit + CI) and `scripts/release_gate.py` (6 docs, overlapping)
  — plus ~6 separate unit tests, using a mix of byte-exact-markdown compare (the
  brittle pattern that just reddened CI), semantic cross-check, and in a couple
  of cases no regeneration gate at all. Only 2 of the dashboards have a
  machine-readable CSV canonical; there is no single "regenerate every generated
  doc" command (the new `--write` covers only the 7 registered ones); and the
  generator CLIs use inconsistent write/check flags (`--render` / `--write` /
  default-write / `--surface=` / `--target=`, with `--out` / `--output`).

## Next Work

1. Add `component_ops`, `fusion_groups`, `shape_envelope`, `effects`, and
   `layout_contracts` to canonical compile metadata. **`component_ops` landed
   2026-06-02**; `fusion_groups` / `shape_envelope` / `effects` /
   `layout_contracts` remain.
2. ~~Gate whole programs and component ops separately.~~ **Landed 2026-06-02**
   — `program_executable` + `component_blockers` gate the whole program
   component-by-component alongside the primary-op `executable` answer.
3. Make Target IR emit backend descriptors rather than embedding/rediscovering
   large Apple-specific fusion/runtime decisions.
4. Require fixture-backed numerical proof before conformance cells become
   complete.
5. Update specs to point at dashboards and this audit instead of old root audit
   documents.
6. **Unify generated-doc regeneration + drift into one contract.** Define a
   single registry of `(doc, generator, check-cmd, write-cmd)` consumed by both
   the CI gate and one `--write` regenerator so finishing a sprint regenerates
   *every* generated doc in one command; fold `release_gate.py`'s drift checks
   into the same registry (retire the second parallel entry point); standardize
   the generator CLI on `--check` / `--write` / `--out`; and extend the
   CSV-canonical + non-byte-gated-Markdown pattern to the remaining data-shaped
   dashboards (`support_table`, `runtime_execution_matrix`, `e2e_op_coverage`,
   `test_coverage_by_op`, `op_target_conformance`, `s_series_status`,
   `standalone_primitive_coverage`, `tsol_coverage`) so machine consumers and
   CI both diff a stable CSV instead of whitespace-fragile Markdown.

## Source Material Consolidated

- `archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`
- `archive/compiler_correctness_testing_audit.md`
- `archive/compiler_improvement_milestone_plan_2026_05_18.md`
- `archive/compiler_layer_gap_remediation.md`
- `archive/compiler_spec_gap_audit.md`
- `archive/docs/audit/compiler/COMPILER_AUDIT.md`

