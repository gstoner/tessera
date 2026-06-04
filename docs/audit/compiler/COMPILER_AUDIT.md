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
- **Generated-doc regeneration + drift gating — registry landed (2026-06-04),
  family-collapse consolidation still open.** The fragmentation finding (two
  parallel gate scripts + piecemeal unit gates + inconsistent generator CLIs)
  has been mostly addressed: `python/tessera/compiler/generated_docs.py` is now
  the single registry of all 21 fully-generated dashboards; `check_generated_docs.sh`
  and `release_gate.py` both delegate to it (the second entry point's per-doc
  drift gates were folded into one fleet-wide `generated_docs_drift`); a unified
  `--write` regenerates the whole fleet; and the fleet drift test
  `tests/unit/test_generated_docs_registry.py` includes an orphan guard so a new
  dashboard must register. The registry immediately caught 3 silently-stale
  dashboards (`test_coverage_by_op`, `test_coverage_classification`,
  `effect_lattice_audit`). **6 dashboards are now CSV-canonical** (`runtime_abi`,
  `verifier_coverage`, `support_table`, `op_target_conformance`,
  `runtime_execution_matrix`, `test_coverage_by_op`). *Still open:* CSV-canonical
  for the remaining data-shaped docs (`test_coverage_classification`,
  `tsol_coverage`, `effect_lattice_audit`, `apple_target_map`, the two
  `*_target_map`s), and the **aggressive content consolidation** (collapse the 3
  target maps → 1 multi-target doc; the 5 surface-status docs → 1; merge the
  test-coverage pair; fold `operator_benchmarks_coverage` into benchmarks; fold
  the `e2e_op_coverage` + `s_series_status` rollups into their primaries) — each
  is now a localized registry edit + generator change.

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
6. **Unify generated-doc regeneration + drift into one contract — landed
   2026-06-04.** `tessera.compiler.generated_docs` is the single registry
   consumed by both `check_generated_docs.sh` and `release_gate.py` (the latter's
   per-doc drift gates folded into one fleet-wide `generated_docs_drift`), with a
   fleet `--write`/`--check`, an orphan-guard test
   (`tests/unit/test_generated_docs_registry.py`), and a `--list` view.
   - **9 dashboards CSV-canonical:** `runtime_abi`, `verifier_coverage`,
     `support_table`, `op_target_conformance`, `runtime_execution_matrix`,
     `tsol_coverage`, `effect_lattice_audit`, plus the merged `test_coverage` and
     consolidated `surface_status`.
   - **Content consolidation done (genuinely-duplicative docs):** the 5
     surface-status docs + `operator_benchmarks_coverage` → one `surface_status`
     (6→1); `test_coverage_by_op` + `test_coverage_classification` → one
     `test_coverage` (2→1). Registry count 24 → 15.
   - **Deliberately not consolidated (reassessed):** the 3 target maps stay
     per-platform — they are *not* duplicative (per-target capability matrices),
     have heterogeneous schemas, and are cross-referenced by 8 per-platform audit
     docs (`backend/{apple,nvidia,rocm}/`); collapsing them would fight the
     per-platform audit structure for a 3→1 saving. The `e2e_op_coverage` /
     `s_series_status` rollups likewise stay standalone — they are distinct
     MASTER_AUDIT truth views, and the registry already removed the duplication
     that mattered (one regen/drift contract). Folding them is available if
     desired but is low-value churn now.

## Source Material Consolidated

- `archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`
- `archive/compiler_correctness_testing_audit.md`
- `archive/compiler_improvement_milestone_plan_2026_05_18.md`
- `archive/compiler_layer_gap_remediation.md`
- `archive/compiler_spec_gap_audit.md`
- `archive/docs/audit/compiler/COMPILER_AUDIT.md`

