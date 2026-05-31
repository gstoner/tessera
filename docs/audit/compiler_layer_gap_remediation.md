# Compiler-Layer Gap Remediation Plan

> Status: **active** (started 2026-05-31). Sequenced per the highest-leverage fix
> order. Each item is a self-contained unit: it lands with tests + an honest
> verification note, and — per the project's discipline — if a real
> implementation isn't tractable, the *claim* is downgraded to match reality
> rather than left overstated.

## Why this exists

The four-layer IR stack (Graph → Schedule → Tile → Target) + runtime has several
gaps where a layer is **documented as implemented but is actually a skeleton,
heuristic, or inspection-only artifact**. The cross-cutting principle for every
fix below: **a single source of truth, and claims that match it.** Compiler
artifacts, the capability registry, the Python runtime, the C ABI, and the docs
must all agree on what a target can actually do.

## Gap inventory (grounded to code)

| # | Layer | Concrete gap | Evidence |
|---|-------|--------------|----------|
| G1 | Schedule→Tile | `_lower_schedule_ops` **drops** `schedule.mesh.define` / `schedule.layout` / `schedule.artifact` (`continue`), so placement/movement semantics never reach Tile IR | `tile_ir.py` `_lower_schedule_ops` |
| G2 | Schedule→Target (C++) | `LowerScheduleToTargetPass::runOnOperation()` is an **empty body** (one comment) but is registered as a pass | `LowerScheduleToTarget.cpp` |
| G3 | Tile→Apple (C++) | C++ Apple pass emits broad `artifact_only` Metal/MPS contracts; the Python/runtime Metal-4 surface has moved far ahead | `TileToApple.cpp` |
| G4 | Execution truth | Capability registry, `runtime.launch()`, C ABI, and compiler artifacts don't derive a runtime-execution matrix from **one** source | `capabilities.py`, `runtime.py` launch |
| G5 | C ABI | `tsrCompileArtifact` / `tsrLoadArtifact` / `tsrGetKernel` / launch return `TSR_STATUS_UNIMPLEMENTED` | `tessera_runtime.cpp` |
| (bg) | Frontend / Graph IR | dynamic Python control flow → unsupported diagnostics; numeric-policy/value-kind facts are optional carriers | `graph_ir.py`, `parser.py` |

## Fix order + acceptance criteria

### 1. Schedule→Tile: preserve mesh / layout / artifact metadata — ✅ DONE (2026-05-31)
**Landed:** `_lower_schedule_ops` now emits `tile.mesh.define` / `tile.layout` /
`tile.artifact` and wraps the mesh region in `tile.mesh.region` (carrying
`mesh`+`axis`) instead of dropping them; `target_ir._flatten_tile_ops` filters
`TILE_METADATA_OPS` (skips the marker, recurses the region body) so they don't
leak into Target lowering as spurious compute. Test
`test_schedule_to_tile_preserves_placement_metadata`; 1052 IR/spine/apple tests
green, zero regressions.

- **Now:** the three metadata op kinds are silently discarded; Tile IR can't
  explain *where* data lives or *how* it moves.
- **Target:** lower them to first-class Tile metadata ops (`tile.mesh.define`,
  `tile.layout`, `tile.placement`) and carry the enclosing-mesh association onto
  the ops inside `schedule.mesh.region`, so Tile IR is a real placement/movement
  bridge, not just an inspection skeleton.
- **Accept:** a Graph→Schedule→Tile round-trip preserves every mesh/layout name;
  Tile IR verifier accepts the new ops; a test asserts the metadata survives and
  is attached to the right tile compute ops. No behavioral regression in existing
  tile lowering tests.

### 2. C++ `LowerScheduleToTargetPass`: implement minimally or stop claiming it — ✅ DONE (2026-05-31, honest downgrade)
**Landed (downgrade path):** the pass was pseudocode, **never registered** in
`tessera-opt`/any pipeline, and redundant with the *tested* Python
Schedule→Tile→Target spine. Rather than ship an untested, unused C++
reimplementation (itself an overstated-claim gap), the pass body now **fails
loudly** (`emitError` + `signalPassFailure`) instead of silently succeeding as a
no-op, so it can never masquerade as a real lowering; it also carries an honest
`getArgument`/`getDescription`. Added a `compiler_spec_gap_matrix.md` row
classifying it **`stubbed`** with the real path (Python spine) named. C++ idioms
match existing passes (`VerifyTesseraIR` `m.emitError`, `AutodiffPass`
`signalPassFailure`); not compile-tested here (no MLIR build in the Python env).
207 IR/spine tests green; no Python dependency on the pass.

- **Target (preferred):** a minimal *real* pass — map `schedule.warp`/pipeline/
  policy to async regions + barriers (`persistent` → 1 CTA/SM), even if narrow.
- **Fallback (honest):** if a real pass is out of scope this pass, mark it
  explicitly scaffolded (rename/annotate) and remove it from "implemented"
  claims + the registered-pass list, with a lit/test note.
- **Accept:** either a lit fixture shows a non-trivial transform, OR the docs +
  pass registry no longer assert it's implemented.

### 3. Tile→Apple (C++) align with the Python/runtime Metal-4 surface — ✅ DONE (2026-05-31)
**Landed:** `TileToApple.cpp`'s GPU pass now tags each lowered
`tessera_apple.gpu.metal_kernel` with `status = "metal_runtime"` for the **full**
runtime envelope — `isAppleGpuRuntimeOp` mirrors
`driver._APPLE_GPU_{MPS,MSL,MPSGRAPH}_OPS` (27 ops: MPS matmul/gemm/batched_gemm,
MSL rope/flash_attn/softmax[_safe]/gelu, MPSGraph Tier-1 activations/norms) — and
`"artifact_only"` for everything else (was a blanket `artifact_only`).
**Compile-verified + lit-verified** against the real MLIR 21.1.8 build:
`apple_gpu_lowering.mlir` updated (matmul/softmax/rope/flash_attn →
`metal_runtime`, generic op → `artifact_only`) + 2 other phase8 Apple fixtures
still pass. **Single-source enforcer:** `test_apple_gpu_tile_pass_status_matches_envelope`
runs the real pass over every `driver` envelope op and fails if the C++ list and
the Python envelope ever diverge.

- **Target:** the C++ Apple pass emits target ops that reflect the *actual*
  runtime envelope (MPS / MSL / MTL4 cooperative / MPSGraph), not a blanket
  `artifact_only` Metal contract — driven by the same envelope `driver.py` uses.
- **Accept:** lit fixtures show the Apple pass distinguishing runtime-executable
  ops from artifact-only ones, consistent with `driver._APPLE_GPU_*` + the
  capability registry.

### 4. Single-source runtime execution matrix
- **Target:** one registry function that, given (target, op, dtype), returns the
  authoritative execution status (`ready` / `metal_runtime` / `artifact_only` /
  `unimplemented`), consumed by `runtime.launch()`, `capabilities.py`, and a
  generated dashboard. `launch()` stops hard-coding the non-CPU "unimplemented"
  branch and instead consults the matrix.
- **Accept:** a generated `docs/audit/runtime_execution_matrix.md` + a drift test;
  `runtime.launch()` and `capabilities.py` both read the matrix; the Apple
  executable envelopes are represented as data, not special-cased code.

### 5. C ABI artifact compile / load / get-kernel / launch
- **Target:** wire the `tsr*Artifact` lifecycle for at least the CPU backend
  (compile a Graph/Tile IR artifact → loadable object → `tsrGetKernel` →
  `tsrLaunchKernel`), end-to-end, with the other backends returning honest
  `UNIMPLEMENTED` until their codegen lands.
- **Accept:** a C ABI smoke test compiles + launches a trivial CPU kernel through
  the `tsr*Artifact` path; non-CPU paths still return `UNIMPLEMENTED` with a
  precise reason (no silent zeros).

## Working agreement
- One item at a time, top-down; each lands tested + measured.
- Prefer the honest downgrade over an overstated claim (G2's fallback is the
  template).
- Every item ends by reconciling the relevant doc/registry so the "single source
  of truth" invariant holds.
