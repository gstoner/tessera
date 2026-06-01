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

### 11. Audit follow-up bug fixes — ✅ DONE (2026-05-31)
Three bugs surfaced by an external code-review pass over A / B / D and
verified empirically before fixing:

* **P1 — AugAssign Sub/Div silently dropped the op.** ``visit_AugAssign``
  accepted ``ast.Sub`` and ``ast.Div`` as part of D.4, but
  ``_try_map_binop`` only knew ``ast.Add`` and ``ast.Mult``. ``c -= b``
  therefore produced the ``PY_FRONTEND_AUGASSIGN_DESUGARED`` info note,
  ``_emit_expr`` returned ``None`` on the resulting ``Sub`` BinOp, and
  **no op was emitted** — silently dropped. **Fix**: added real
  ``tessera.sub`` / ``tessera.div`` lowering symmetric with add/mul, and
  widened the ``_emit_expr`` BinOp accept-set. Both AugAssign forms AND
  plain ``a - b`` / ``a / b`` BinOps now lower to real ops.
* **P2 — pipeline_gates rocm_gfx* per-arch targets had no manifest
  match.** ``_manifest_entries`` mapped family→per-SM for NVIDIA but had
  no inverse rule for ROCm (which emits ``target="rocm"`` in the
  manifest, not per-arch). A per-arch target like ``rocm_gfx942``
  matched zero manifest rows and the codegen gate spuriously failed
  *before* the intended ``toolchain`` (no hipcc) gate. **Fix**: added
  the symmetric per-arch ROCm rule — ``rocm_gfx*`` inherits the
  ``rocm`` family manifest row. Now ``first_failing_gate("rocm_gfx942",
  "matmul")`` correctly returns ``toolchain`` with the "no hipcc"
  detail, matching the family-target answer.
* **P2 — _platform_is_darwin_arm64 only checked Darwin, not arm64.**
  The helper name promised arm64 but the body was
  ``sys.platform == "darwin"``. On an Intel Mac this would have falsely
  passed the apple_cpu / apple_gpu ``hardware_smoke`` gate. **Fix**:
  also check ``platform.machine() == "arm64"``. The
  ``test_apple_gpu_hardware_smoke_fails_on_intel_mac`` regression guard
  patches the helper's environment to simulate the Intel Mac path and
  confirms the gate now reports ``fail``.

15 new tests in
``tests/unit/test_audit_fixes_d5_b_platform_b_rocm.py`` (3 AugAssign
guards covering sub/div/plain BinOps, 7 ROCm guards covering all five
per-arch targets + the family target + NVIDIA preservation, 5 platform
guards covering Intel Mac / arm64 Mac / Linux / the downstream
``hardware_smoke`` gate). **1003 affected tests pass** across the wide
sweep (jit / canonical / compile / graph_ir / control_flow / dynamic /
runtime / pipeline / conformance / execution_matrix / audit_fixes), 4
skipped, 0 failures, 0 regressions.

### 10. Dynamic control flow lowering — ✅ D.1 + D.2 + D.3 DONE (2026-05-31)
**Audit recommendation D.** The graph_ir frontend (line 922 pre-D) emitted a
blanket ``PY_FRONTEND_UNSUPPORTED`` warning for every dynamic ``if`` /
dynamic ``for`` / ``while`` / aug-assign and produced **no IR markers
at all**. The audit's framing: "That is a big gap for real model code
unless tracing/AOT capture becomes the canonical frontend."

D closes the gap by lowering all four to the same ``tessera.scf.*``
marker shape the static path already uses — downstream consumers see a
structural boundary regardless of whether the specific test / iterable
/ condition expression is yet emittable as SSA.

**Three-tier lowering contract.** Each dynamic construct goes through:

1. **Static path** (unchanged) — ``ast.literal_eval`` resolves the test
   to a bool or ``range(N)`` resolves to a constant. The markers carry
   the resolved value as an attribute (``condition=True``,
   ``trip_count=3``). No diagnostic.
2. **Dynamic + lowerable** — ``_emit_expr`` produces an SSA value for
   the test / trip-count. The marker carries it as an *operand* with
   ``kind="dynamic"`` and emits an info note
   (``PY_FRONTEND_DYNAMIC_IF_LOWERED`` / ``..._FOR_LOWERED`` /
   ``PY_FRONTEND_WHILE_LOWERED``).
3. **Dynamic + not yet emittable** (e.g. ``ast.Compare``, attr-call
   chains the frontend doesn't yet lower) — the marker still emits with
   the Python source recorded as ``condition_text=`` /
   ``iter_text=``, plus a more specific info note
   (``PY_FRONTEND_DYNAMIC_IF_UNLOWERED_CONDITION`` /
   ``..._FOR_UNLOWERED_ITERABLE`` /
   ``PY_FRONTEND_WHILE_UNLOWERED_CONDITION``). The downstream consumer
   sees the structural shape and a precise hint about what's missing.

**Aug-assign bonus (D.4).** ``x += y`` (and ``-= *= /=`` with a plain
``Name`` target) now desugars to ``x = x op y`` and goes through the
normal Assign + BinOp path, with an info note
``PY_FRONTEND_AUGASSIGN_DESUGARED``. Subscript targets like
``a[i] += b`` stay honestly unsupported (one short note in the same
test file).

**Source-text fidelity.** ``_safe_unparse(node)`` records up to 120
chars of the original Python expression on the marker, so an IR dump
shows what the user wrote even when the AST node isn't yet lowered. A
human reading IR sees ``condition_text = "x.sum() > 0"`` instead of an
opaque structural boundary.

**Regression guards.** Two specific guards lock the D contract:
* ``test_pre_d1_unsupported_warning_no_longer_fires_for_dynamic_if`` —
  the literal pre-D.1 warning string must not appear for any dynamic
  ``if`` / ``for`` / ``while`` body.
* ``test_unsupported_python_construct_has_source_span_diagnostic`` in
  the legacy ``test_graph_ir.py`` was updated to assert the *new*
  contract (``PY_FRONTEND_DYNAMIC_IF_UNLOWERED_CONDITION`` info note +
  structural ``tessera.scf.if.*`` markers + source span preserved).

**Honest scope.** D closes the *frontend lowering* gap. The downstream
CPU compiler refuses to plan a body containing ``tessera.scf.if.begin``
("op is not supported by the CPU compiler path") and falls back to
eager Python — that's correct B.2 behavior, surfaced by a precise
``JIT_EAGER_FALLBACK_UNSUPPORTED_OP`` diagnostic. Making
``tessera.scf.if`` *executable* through a backend (e.g. via a real
``scf.if`` lowering pass) is a separate, larger surface tracked
separately. D's contribution is: the frontend no longer pretends a
construct doesn't exist, and the downstream layer gets a structural
shape it can implement against when ready.

11 D-focused tests in ``tests/unit/test_dynamic_control_flow_lowering.py``
plus the updated ``test_control_flow_lowering.py`` /
``test_graph_ir.py`` lock the contract. **988 affected tests pass**
across jit / canonical / compile / graph_ir / control_flow / dynamic /
runtime / pipeline / conformance / execution_matrix
(4 skipped, 0 failures, 0 regressions).

### 9. Canonical compiler driver — 🟡 C.1 DONE (2026-05-31)
**Audit recommendation C** (started after B). The goal:

    Make one canonical compiler driver own the whole ladder: Graph module
    in, typed artifacts plus execution capability out. Right now driver.py,
    matmul_pipeline.py, backend manifests, target maps, and runtime
    dispatch each hold part of the truth.

**Landed (C.1) — canonical wrapper + tests + cross-target demo.**

- New pure-aggregator module
  ``python/tessera/compiler/canonical_compile.py`` exposes
  ``canonical_compile(module, *, target) → CompileResult`` — the audit's
  ``compile(module, target) → (typed_artifacts, capability_set,
  executable | reason)`` contract.
- ``CompileResult`` is a frozen dataclass with:
  * Typed artifact accessors ``graph_ir`` / ``schedule_ir`` /
    ``tile_ir`` / ``target_ir`` mirroring the bundle text.
  * ``gate_results`` — full seven-gate table from
    ``pipeline_gates.evaluate``.
  * ``first_failing_gate`` — the audit-named gate (or None).
  * ``executable`` — AND of bundle.executable and "no gate fails."
  * ``reason`` — empty when executable; leads with the audit-named gate
    diagnostic otherwise (``"first failing gate `<name>` — <detail>"``).
  * ``to_dict()`` — stable surface for downstream telemetry / dashboards.
- Pure aggregator: only imports from ``driver``, ``graph_ir``, and
  ``pipeline_gates`` (the upstream truth sources). Locked by
  ``test_module_is_pure_aggregator``.
- **Cross-target demo** (one ``canonical_compile`` call per target):

  | target          | executable | first failing gate | detail |
  |-----------------|------------|---------------------|--------|
  | `cpu`           | ✅ True    | —                   | (all gates pass) |
  | `apple_cpu`     | ✅ True    | —                   | (all gates pass) |
  | `apple_gpu`     | ✅ True    | —                   | (all gates pass) |
  | `nvidia_sm90`   | ❌ False   | `toolchain`         | nvcc not on PATH |
  | `rocm`          | ❌ False   | `toolchain`         | hipcc not on PATH |
  | `metalium`      | ❌ False   | `link`              | artifact_only |

- 12 structural + behavioral tests in
  ``tests/unit/test_canonical_compile.py``: typed-artifact round-trip,
  gate-result agreement with the direct evaluator, executable/reason
  agreement on known-good and known-failing cases, pure-aggregator
  import allowlist, primary-op extraction (with ``tessera.`` prefix
  stripping), and a "one surface carries the whole answer" guard.

**C.2 — runtime.launch() trusts CompileResult — ✅ DONE (2026-05-31).**

- Added ``CompileResult.to_runtime_artifact()`` — projects the canonical
  answer (executable / reason / first_failing_gate / gates / IR text)
  into a ``RuntimeArtifact`` via a stable set of ``canonical_*``
  metadata keys, plus ``abi_signature = "tessera.canonical.v1.<target>"``
  so downstream consumers can detect canonical provenance.
- Taught ``runtime._first_failing_gate_for_metadata`` to **trust** the
  canonical answer when present: if ``canonical_first_failing_gate`` is
  in metadata, it builds a ``GateResult`` from the stamped fields and
  returns without calling ``pipeline_gates.first_failing_gate``. Same
  truth, zero re-derivation. Legacy artifacts (no ``canonical_*`` keys)
  still flow through the B.2 derive path unchanged.
- 9 tests in ``tests/unit/test_canonical_to_runtime.py``: round-trip
  shape guards, IR round-trip, abi_signature, runtime trust path for
  nvidia / rocm / metalium, legacy fall-through, and a "no first failing
  gate when every gate passes" guard.

**C.3 — @jit carries the canonical CompileResult — ✅ DONE (2026-05-31).**

- ``@tessera.jit`` now synthesizes a ``CompileResult`` from the bundle
  it already built (no second compile — uses
  ``compile_result_from_bundle``, which runs only the gate evaluator
  over the existing ladder output).
- ``JitFn`` gained ``compile_result: Optional[CompileResult]``; legacy
  fields (``compile_bundle`` / ``cpu_plan`` / ``execution_kind``) are
  unchanged. ``jit_fn.compile_result.bundle is jit_fn.compile_bundle``
  by construction — one bundle, two views.
- **The audit-named gate is now resolved at *decoration* time.**
  ``@jit(target="nvidia_sm90")`` on a host without CUDA returns a
  ``JitFn`` whose ``compile_result.first_failing_gate.gate ==
  "toolchain"`` — the user can inspect the diagnostic before any call.
- 7 tests in ``tests/unit/test_jit_canonical_compile.py``: presence,
  bundle-identity, target match, seven-gate carry, decoration-time
  diagnostic for nvidia, legacy fields preserved, ``to_dict`` round
  trip on ``JitFn``.

**Status: C complete.** ``canonical_compile`` is the canonical contract;
``runtime.launch`` consumes it; ``@tessera.jit`` produces it. The five
today-owners (``driver`` / ``matmul_pipeline`` / ``backend_manifest`` /
``execution_matrix`` / runtime dispatch) keep their truth, but every
consumer reaches them through one place.

**Helpful side effect surfaced by C.3:** previously a function like
``def f(a, b): return a @ b`` decorated with ``@jit`` would silently fall
back to eager Python with no explanation. Now ``f.compile_result.reason``
carries ``"bundle reports non-executable artifact: runtime_status=unsupported
execution_kind=fallback_eager"`` — the user sees *why* the fast path
didn't apply. (Adding a fusion-pass or Graph IR lowering for the
``MatMult`` AST node will flip this to executable; the diagnostic now
points at the right place to fix.)

**855 affected tests pass across every jit/compile/canonical/runtime
test in the repo** (4 skipped, 0 failures) — proves C.1+C.2+C.3 are
fully additive.

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

**B.2 — runtime.launch() integration — ✅ DONE (2026-05-31).** Both
unsupported paths in ``runtime.launch()`` (the non-CPU branch and the
CPU fall-through) now call
``pipeline_gates.first_failing_gate(target, op_name)`` and:

- Lead the ``reason`` string with the audit-named gate:
  ``"unsupported: first failing gate `toolchain` — nvcc not on PATH
  (CUDA Toolkit 13.2.1 not installed). (see
  docs/audit/op_target_conformance.md)"``.
- Carry top-level structured keys ``first_failing_gate`` and
  ``first_failing_gate_detail`` on the response dict (machine-readable;
  callers don't have to parse the reason string).
- Mirror the same keys into ``telemetry.metadata`` so downstream
  consumers of the runtime telemetry stream see the named gate.
- Resolve sub-target names (``nvidia_sm90`` / ``rocm_gfx942`` / etc.)
  to the family-level gate via ``pipeline_gates._normalize_target`` so
  per-arch artifacts get the same diagnostic as their family aggregate
  in the dashboard.

The unsupported ``runtime_status`` enum (``unimplemented`` /
``missing_backend`` / ``unsupported``) stays unchanged — adding the
gate name is an *additional* honest channel, not a replacement.
Locked by ``tests/unit/test_runtime_launch_first_failing_gate.py``
(7 tests: 3 named cases for nvidia / rocm / metalium, 1 structured-key
shape guard, 1 status-stays-honest guard, 1 cross-link guard, 1 op-name
extraction guard).

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
