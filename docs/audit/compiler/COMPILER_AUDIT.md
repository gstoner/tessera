# Compiler Audit

This document consolidates the compiler audit material that previously lived in
multiple root audit documents and compiler archive files.

> **Latest deep pass:** [DEEP_COMPILER_AUDIT_2026_06_10.md](DEEP_COMPILER_AUDIT_2026_06_10.md)
> — source-backed audit of frontend/IR/manifest/runtime-ABI/Apple-envelope/
> benchmark coverage. Records the "generated drift clean vs semantic gap open"
> split, fixes the bench-axis staleness + the grouped_gemm/moe_swiglu_block
> manifest blind spot, and carries a prioritized gap table for the rest.
>
> **Code-level companion:** [CODE_AUDIT_2026_06_10.md](CODE_AUDIT_2026_06_10.md)
> — refactoring / per-IR-level optimization correctness / glass jaws. Headline:
> a verified `TransposeIntoMatmul` flag-composition miscompile (fixed, commit
> `acb5c6f`), missing fusion use-guards (fixed), NSA gating-semantics hazard
> (guarded), silent autodiff chain breaks (diagnosed), no upstream
> canonicalizer/CSE in named pipelines (fixed), `TESSERA_STRICT_DISPATCH`
> against silent numpy fallbacks, and runtime consumption of `fusion_groups`.
> Two earlier agent claims refuted.
>
> **Evaluator program — substantially shipped (2026-06-12):** [EVALUATOR_PLAN.md](EVALUATOR_PLAN.md)
> (see its §9.5 "what has landed"). A generative, execution-derived,
> **backend-rung-aware** Evaluator that *derives* conformance/benchmark/autotune
> surfaces from one honest scoring engine (closing the "registry models reality"
> gap). **Landed:** the 8-rung verdict engine + provenance gate
> (`evaluator.py`); four oracles — vertical, horizontal/PolyJuice, metamorphic,
> and DESIL cross-path (`cross_path_equivalence`); conformance corroboration
> (`conformance_evaluator.py`); the autotuning flywheel + per-chip calibration +
> autotune_v2 bridge (`flywheel.py`, `flywheel_autotune.py`); and the scored
> environment — a TensorBench-style grader (`compiler_grader.py`), LongCA
> structure-keyed attention (`attention_tasks.py`), and Magellan/AlphaEvolve
> gated search (`magellan.py`, `alphaevolve.py`, with reward-hack rejection
> proven). **Open (hardware-gated):** NVIDIA/ROCm sit truthfully at rung 1–2.5
> (WGMMA PTX *emitted* via `ptx_emit.py`; rung-3 `ptxas` + complete kernel +
> silicon need a Linux/CUDA runner). Research-backed
> (DESIL/PolyJuice/Mirage/TensorBench/Magellan/AlphaEvolve/BaCO/TLP).

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
  component-aware metadata landed (2026-06-07).** `CompileResult` carries
  ``component_ops`` (the whole-program distinct op vocabulary),
  ``program_executable`` (gated component-by-component, not just the primary
  op), and ``component_blockers`` ((op, failing-gate) pairs). **`effects` /
  `shape_envelope` / `layout_contracts` / `fusion_groups` now reach the
  user-facing `fn.runtime_artifact().metadata`** — derived in
  `canonical_compile._derive_*`, factored into
  `CompileResult.descriptive_metadata()`, and merged additively through
  `JitFn._build_runtime_artifact` (previously discarded on the `@jit` path —
  every key was absent for real jitted functions). `fusion_groups` recognizes
  the cross-family chains the Apple GPU runtime actually fuses
  (`matmul→softmax[→matmul]`, `matmul→gelu`, `matmul→rmsnorm`), not just
  same-family adjacency. Locked by `tests/unit/test_canonical_component_ops.py`
  + `tests/unit/test_canonical_metadata_jit.py`. **Graph outputs landed
  2026-06-11** (`canonical_outputs` + populated `return_values`/`result_types`;
  see Next Work #1). **Runtime consumption of `fusion_groups` landed
  2026-06-10** (see next item).
- **Fusion intent is too late — runtime half closed (2026-06-10).** The
  apple_gpu executor now consults `fusion_groups` known_chain metadata before
  the structural re-matchers (which remain as legacy-artifact fallback);
  locked by `tests/unit/test_strict_dispatch.py` (short-circuit + legacy-path
  tests). **SwiGLU is now derived too** (`_match_swiglu_at` handles the DAG —
  gate/up share %x — inside the known-chain scan) and consumed by the
  executor. **Target IR descriptor consume/emit — landed 2026-06-11.** All 7
  Apple fusion passes now *emit* a first-class fusion descriptor on the fused
  call (`tessera.fusion.kernel` + `tessera.fusion.source`): the 4 chain passes
  (matmul→softmax→matmul / matmul→softmax / matmul→gelu / matmul→rmsnorm) also
  *consume* an upstream `tessera.fusion.intent` (source `"descriptor"` vs
  `"rediscovered"`, with a Decision-#21 warning on descriptor/IR disagreement);
  the 3 composite passes (swiglu / mla_decode / native_sparse_attn) emit
  `source = "composite_op"` (the pre-fused op *is* the descriptor). The Python
  emit-half `canonical_compile.stamp_fusion_intents(module)` stamps the intent on
  each recognized chain's terminal op from the canonical `_KNOWN_FUSION_CHAINS`,
  so the frontend produces descriptor-annotated IR. Lit:
  `tests/tessera-ir/phase8/apple_gpu_fusion_descriptor.mlir`; Python:
  `tests/unit/test_apple_fusion_descriptor.py` + `test_fusion_intent_emitter.py`
  (incl. an emit↔consume contract guard). **Auto-wired 2026-06-11:**
  `driver.compile_graph_module` calls `stamp_fusion_intents(module)` before
  rendering the Graph IR for Apple targets (gated to `apple_gpu`/`apple_cpu`;
  the descriptor is backend-agnostic so it extends when other backends consume
  it), so every Apple compile now produces descriptor-annotated Graph IR that
  the Target IR passes consume. The intent is stamped into the op's MLIR `attrs`
  (not `kwargs`, which are the op's real call arguments in the reference/runtime
  path). Loop closed end-to-end.
- **Layout and binding contracts are uneven.** Graph/Schedule/Tile/Target IR
  need stronger dtype, layout, aliasing, and buffer-binding contracts.
  **Layout slice extended 2026-06-11:** `LayoutLegalityPass` was matmul-only; its
  producer/consumer accept-set rule now also covers `tessera.conv2d_nhwc` (nhwc
  on the data operand #0; the filter is a separate weight layout) and
  `tessera.flash_attn` (bhsd on Q/K/V #0..2), per-operand-scoped so it only
  checks the operands that carry each contract. matmul stays verbatim (the V4a
  diagnostic + `matmulAcceptSet()` are pinned by existing tests). Lit:
  `tests/tessera-ir/phase2/layout_conv_flashattn_accept_set.mlir`; Python:
  `tests/unit/test_layout_legality_extended.py`. Still open: dtype / aliasing /
  buffer-binding contracts, and wiring `LayoutLegalityPass` into the named
  pipelines (it's still registered standalone).
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
  `effect_lattice_audit`). **CSV-canonical data-shaped tail closed 2026-06-11 —
  12 dashboards now CSV-canonical:** `runtime_abi`, `verifier_coverage`,
  `support_table`, `op_target_conformance`, `runtime_execution_matrix`,
  `test_coverage`, `tsol_coverage`, `effect_lattice_audit`, `surface_status`, and
  the **3 target maps** (`apple_target_map` + `nvidia_sm90_target_map` +
  `rocm_target_map`, added via `apple_target_map.render_csv` /
  `gpu_target_map.render_csv(target)`, wired into the registry so the CSV is the
  drift-gated artifact). The remaining markdown-only docs are narrative rollups
  (`e2e_op_coverage`, `s_series_status`, `s_series_accelerator_proof`,
  `docs_freshness`), not row tables. *Still open (deliberately deferred):* the
  **aggressive content consolidation** (collapse the 3 target maps → 1
  multi-target doc; the `e2e_op_coverage` + `s_series_status` rollups into their
  primaries) — Next Work #6 reassessed these as low-value churn (per-platform
  maps are cross-referenced by the per-platform audit docs; the rollups are
  distinct MASTER_AUDIT truth views).
- **Code-level audit closeout (2026-06-10).** The
  [CODE_AUDIT_2026_06_10.md](CODE_AUDIT_2026_06_10.md) "Closeout status" section
  drove every remaining code-level finding to done / refuted / accepted /
  tracked-deferred. Done: 1e zero-`TRACE_DEFERRED` corpus guard, `_APPLE_GPU_*`
  table-creep enforcer, binary/rowop strict-dispatch funnel coverage, the
  `LoweringUtils.h` dedup across 18 Apple passes, bf16-probe (already cached).
  Explicitly tracked-deferred (with rationale): C-ABI int return code, Target-IR
  C++ fusion-descriptor consumption, Schedule/Tile IR autotuner/LICM (hardware-
  gated), `forbid-ops` pipeline wiring, and the `jit.py` decorator extraction.

## Next Work

1. ~~Add `component_ops`, `fusion_groups`, `shape_envelope`, `effects`, and
   `layout_contracts` to canonical compile metadata.~~ **Landed** —
   `component_ops` (2026-06-02) + `effects` / `shape_envelope` /
   `layout_contracts` / `fusion_groups` (2026-06-07), all reaching the
   user-facing `fn.runtime_artifact().metadata`. **Graph outputs landed
   2026-06-11** — `CompileResult.outputs` / `canonical_outputs`
   (`tessera.compile.outputs.v1`: each returned value + producer op + type /
   shape / dtype / layout), backed by populating `GraphIRFunction.return_values`
   + `result_types` from the jit AST `return` (the AST path previously emitted a
   value-less `return`, so outputs/`shape_envelope.returns` were empty). Locked
   by `tests/unit/test_canonical_outputs.py`; full IR/lit/canonical sweep green.
   Remaining: runtime *consumption* of `fusion_groups` (Next Work #3 / "fusion
   intent too late").
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

