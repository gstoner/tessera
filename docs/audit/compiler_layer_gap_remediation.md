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

### 4. Single-source runtime execution matrix — ✅ DONE (2026-05-31)
**Landed:** new `python/tessera/compiler/execution_matrix.py` is the **one place**
that maps `(target, compiler_path) -> ExecutionRow(execution_kind, executable,
executor_id, runtime_status, reason, execution_mode)`. Four executable rows
today (`apple_cpu/apple_cpu_accelerate`, `apple_gpu/apple_gpu_mps`,
`cpu/native_cpu`, `cpu/jit_cpu_numpy`) + an explicit `_UNIMPLEMENTED_TARGETS`
list (NVIDIA SMs, ROCm GFXs, metalium). `runtime.launch()`'s non-CPU
`unimplemented` fallthrough now cites the matrix (was hard-coded `target != "cpu"
-> unimplemented`); the Apple-CPU / Apple-GPU / CPU branches above it already
match what the matrix says so the matrix is the derivable truth. Dashboard at
**`docs/audit/generated/runtime_execution_matrix.md`** (regenerate via
`python3 -c 'from tessera.compiler.execution_matrix import write_dashboard;
write_dashboard()'`). **Drift test** `test_runtime_execution_matrix.py` (15
tests) enforces: matrix↔capabilities cross-check (every row's target + every
unimplemented target is in `TARGET_CAPABILITIES`); every `KNOWN_EXECUTORS` entry
is used by some row; executable rows have an executor_id (and vice versa);
executable + unimplemented sets are disjoint; lookup semantics; dashboard
byte-equal to live `render_dashboard()`; `runtime.launch()` imports the matrix
and its fallthrough reason cites the dashboard. Caught a real inconsistency
during validation (`'reference_cpu'` was in `KNOWN_EXECUTORS` but no row used it
— it's an internal fallback inside the native_cpu branch, not a directly
dispatched executor; removed from the catalog with a comment). 990 affected
tests green, zero regressions.

- **Target:** one registry function that, given (target, op, dtype), returns the
  authoritative execution status (`ready` / `metal_runtime` / `artifact_only` /
  `unimplemented`), consumed by `runtime.launch()`, `capabilities.py`, and a
  generated dashboard. `launch()` stops hard-coding the non-CPU "unimplemented"
  branch and instead consults the matrix.
- **Accept:** a generated `docs/audit/runtime_execution_matrix.md` + a drift test;
  `runtime.launch()` and `capabilities.py` both read the matrix; the Apple
  executable envelopes are represented as data, not special-cased code.

### 6. GPU artifact lifecycle + target-gated C ABI launch + matrix-as-dispatcher — ✅ DONE (2026-05-31)
**Landed in response to the audit's "what we can mark as real progress" review.**

- **Matrix-as-dispatcher (P1a)**: `runtime.launch()` no longer hard-codes
  `target == "apple_gpu" and compiler_path == "apple_gpu_mps"`. Both Apple-CPU
  and Apple-GPU branches collapsed into a single matrix-driven dispatch that
  consults `execution_matrix.executor_for_metadata(...)`, resolves
  `ExecutionRow.executor_id` against a `_executor_table()`, and runs through one
  unified try/telemetry/result block. Adding a new backend executor is now
  (1) the function + (2) `KNOWN_EXECUTORS` + (3) a matrix row — no fourth
  hard-coded branch.
- **C ABI target tagging (P1b, P2c)**: `tsrArtifact_t` gained `target` /
  `compiler_path` / `execution_kind` fields; payload format bumped to **`TSRART2`**
  (target-tagged) with `TSRART1` still accepted on load for backward compat.
  `tsrKernel_t` became a tagged union (`tsrKernelKind::{kHostCpu, kGpuUnbridged}`).
  New `tsrDestroyKernel` for symmetric ownership.
- **Honest GPU launch (P2b, P2e)**: `tsrLaunchKernel` dispatches by `kind`.
  CPU host kernels still route through `tsrLaunchHostTileKernel` (G5).
  GPU kernels return **`TSR_STATUS_UNIMPLEMENTED`** with the precise reason
  `"no native C-ABI launch bridge for target='apple_gpu' kernel='…'. The
  Python runtime executes this artifact via execution_matrix dispatch; the C
  ABI launch bridge is a separate gap."` — silent success is impossible.
- **Crossover test (P2d)**: new `test_runtime_artifact_abi_g6.py` (4 tests)
  compiles a C harness against `libtessera_runtime.a` and proves the full GPU
  lifecycle: compile (with `options->target="apple_gpu"`) → payload-is-`TSRART2`
  → `tsrLoadArtifact` round-trip preserves the target → `tsrGetKernel` succeeds
  → `tsrLaunchKernel` returns `UNIMPLEMENTED` with the precise reason → v1
  legacy payloads still load. Plus source-level guards on the artifact struct
  fields and tagged-union shape so a regression to G5's CPU-only shape fails
  the test without needing a built runtime.
- **P2a (envelope drift) honest downgrade**: the C++ Apple-GPU envelope list in
  `TileToApple.cpp` is still hand-maintained-with-drift-test (the matmul2d
  pattern). True single-source via codegen/`.td` is a separate item; the drift
  guard in `test_apple_gpu_tile_pass_status_matches_envelope.py` is the
  enforced contract today.

1003 affected tests green; ABI dashboard regenerated (+1 symbol: `tsrDestroyKernel`).

#### 6.1 Matrix-as-dispatcher: CPU rows now route through `_executor_table()` — ✅ DONE (2026-05-31)
Audit residual: G6 left CPU rows (`native_cpu`, `jit_cpu_numpy`) on the older
inline path, so the "matrix is the dispatcher" invariant held only for Apple
CPU/GPU. Closed by adding `apple_cpu_accelerate`, `apple_gpu_mps`,
`native_cpu`, **and** `jit_cpu_numpy` to `_executor_table()` (the last via
`_execute_cpu_native_or_jit`, which preserves the `native_cpu → reference_cpu`
fallback semantics by returning `(output, override_kind)`). The inline CPU
branch in `runtime.launch()` is deleted; every executable row in the matrix is
now reached through exactly one dispatch site.

### 8. Named pipeline capability gates — 🟡 B.1 DONE (2026-05-31)
**Audit recommendation B** (started after A). The goal: replace
``execution_kind: "reference_cpu"`` post-hoc metadata with an ordered,
named gate sequence so the *first un-passed gate* is the diagnostic the
runtime emits, not a single inscrutable kind label.

**Landed (B.1) — gate evaluator module + dashboard column.**

- New pure-aggregator module ``python/tessera/compiler/pipeline_gates.py``
  defines the canonical seven gates: ``legality → codegen → toolchain →
  link → runtime_abi → hardware_smoke → numerical``. Each gate's status
  (``pass`` / ``fail`` / ``not_evaluated`` / ``not_applicable``) is sourced
  from existing truth (capabilities + backend_manifest + execution_matrix +
  primitive_coverage + ``shutil.which`` toolchain probes) — no new private
  truth, drift-guarded by ``test_module_is_pure_aggregator``.
- ``evaluate(target, op_name) → tuple[GateResult,...]`` returns the full
  seven-gate table; ``first_failing_gate(target, op_name)`` returns the
  first ``FAIL`` with its detail string. Every fail carries a non-empty
  reason — locked by ``test_every_failing_gate_has_a_nonempty_detail``.
- **The conformance matrix dashboard now reports the first failing gate
  per (op, target) cell.** Each row in
  ``docs/audit/op_target_conformance.md`` has a new column naming the
  audit-style gate that's blocking. Examples:

  * ``nvidia × matmul``: ``toolchain — nvcc not on PATH (CUDA Toolkit
    13.2.1 not installed)``
  * ``rocm × matmul``: ``toolchain — hipcc not on PATH (ROCm 7.2.3 not
    installed)``
  * ``metalium × matmul``: ``link — artifact_only — IR emits but no
    linked-kernel path today``

- 13 structural + behavioral tests in
  ``tests/unit/test_pipeline_gates.py`` (gate order, known-good pass on
  cpu / apple_cpu / apple_gpu, known-bad first-failing-gate for nvidia /
  rocm / metalium, no-silent-fails, unknown-target/op handled).

**Open (B.2): runtime integration.** When
``@jit(target="nvidia")`` is called on a host without CUDA, the runtime
should refuse with ``"unsupported: first failing gate = toolchain (nvcc
not on PATH)"`` rather than falling through to whatever the matrix
dispatcher decides. The gate evaluator is already factored to make this
a small wiring change; cleanly testing it requires extending the
runtime's error-path coverage and is sequenced separately.

### 7. Op×Target conformance matrix — ✅ DONE (2026-05-31)
**Audit recommendation A** (out of A/B/C/D).

- New module `python/tessera/compiler/conformance_matrix.py` cross-references
  the four existing truth sources (`primitive_coverage` 12-axis contracts,
  `backend_manifest` per-target kernel status, `execution_matrix` runtime
  executors, and the Apple-GPU runtime envelope sets) into a per-(op, target)
  view of the **seven-step proof ladder**: `graph_emitted → schedule_legal →
  tile_legal → target_legal → backend_compile → runtime_execute →
  numerical_check`.
- Dashboard at `docs/audit/op_target_conformance.md` covers 7 ops × 6 targets
  = 42 cells: `{matmul, matmul_relu, softmax, matmul_softmax, conv2d,
  flash_attn, kv_cache_read}` × `{cpu, apple_cpu, apple_gpu, nvidia, rocm,
  metalium}`.
- Pure aggregator — guarded by `test_matrix_is_pure_aggregator` so it can't
  grow private knowledge that shadows the upstream sources. Drift-gated by
  `test_dashboard_in_sync_with_generator`. Eight structural tests total in
  `tests/unit/test_op_target_conformance.py`.
- CLI: `python -m tessera.cli.conformance_matrix --render | --check`.
- **Surfaced 3 real upstream gaps on the first run** (rendered in their own
  dashboard section): `relu`/`conv2d`/`kv_cache_read` on `apple_gpu` are in
  the runtime envelope but missing from `backend_manifest`. Each is a real
  one-line `BackendKernelEntry` addition, kicked back to the manifest as a
  follow-up rather than silently fixed inside the matrix.
- Initial signal: **5 complete / 12 partial / 25 missing**. Honest baseline
  — the dashboard explicitly distinguishes "Graph IR knows about it"
  (`graph_emitted ✅`) from "the runtime can launch a numerically-validated
  kernel today" (overall `complete`). This replaces the audit's
  "architecture-implied vs. executable" framing problem with a single
  drift-gated table.

#### 6.2 Public `tsrGetArtifactBytes` / `tsrGetArtifactTarget` ABI — ✅ DONE (2026-05-31)
Audit residual: the G6 test inspected the opaque artifact payload via a
private layout-mirror struct that had to be kept in sync with the C++ side by
hand. Closed by adding two read-only view functions to the public header
(`tessera_runtime.h`): `tsrGetArtifactBytes(artifact, &bytes, &len)` returns
the canonical serialized payload (the same bytes `tsrLoadArtifact` accepts),
and `tsrGetArtifactTarget(artifact, &target)` returns the target tag. Both
the G5 round-trip harness and the G6 GPU-lifecycle harness were rewritten to
use the public API — no test now pokes at `tsrArtifact_t`'s internals. The
runtime ABI dashboard accepts both new symbols (drift gate clean).

### 5. C ABI artifact compile / load / get-kernel / launch — ✅ DONE (2026-05-31, CPU end-to-end)
**Landed:** the four `tsr*Artifact` lifecycle functions that returned
`TSR_STATUS_UNIMPLEMENTED` now wire end-to-end for the CPU backend, on top of
the working `tsrLaunchHostTileKernel`. New `tsrRegisterHostKernel(name, fn)`
puts a CPU host kernel into a process-wide registry;
`tsrCompileArtifact("<name1>,<name2>,...", opts, &out)` bundles them with a
canonical text payload (`TSRART1` magic + sorted `name\tfn_ptr` table);
`tsrLoadArtifact(bytes, len, &out)` round-trips the bytes back to a live
artifact and **rejects garbage with `INVALID_ARGUMENT`**; `tsrGetKernel` looks
up by name (returns `NOT_FOUND`, not silent success, for unknown names);
`tsrLaunchKernel(s, k, args, nargs)` interprets `args[0] = const tsrLaunchParams*`
+ `args[1] = void* user_payload` and routes through the existing
`tsrLaunchHostTileKernel`. Unregistered kernels in `tsrCompileArtifact` still
return `UNIMPLEMENTED` (honest — non-CPU codegen JIT is a separate gap).
**Real C++ end-to-end verified:** `tests/unit/test_runtime_artifact_abi.py` (3
tests) compiles two C harnesses against `libtessera_runtime.a` and exercises
(a) the full lifecycle (register → compile → getkernel → launch + correct
`NOT_FOUND`/`UNIMPLEMENTED` semantics) and (b) serialize → `tsrLoadArtifact` →
launch on the re-loaded artifact + garbage rejection; plus a source-level
regression guard that fails if any of the four bodies regresses to a bare
`return TSR_STATUS_UNIMPLEMENTED`. 985 affected tests green, zero regressions.

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
