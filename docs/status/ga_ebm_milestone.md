# GA / EBM milestone status

> **One-page canonical status.** Update this page when something
> changes; everything else in the repo (READMEs, roadmap, audit) cites
> the *claims* below rather than restating them.
>
> **Last updated:** 2026-05-17 (**compiler-integrated vertical slice landed — `@clifford_jit(target="apple_gpu")` produces a traced op plan + manifest-resolved Apple target metadata + benchmark row; rotor-conditioned EBT workload fuses GA + EBM through public APIs**).

## TL;DR

| Surface | Status | Coverage |
|---|---|---|
| **GA primitives (Clifford)** | ✅ all native + benchmarked | 17 / 17 fused MSL kernels on Apple GPU |
| **EBM primitives (native)** | 🟡 close to complete, 1 to go | **8 / 9** with fused MSL kernels |
| **EBM primitives (Python ref only)** | ⏳ 1 op | Only `ebm_partition_exact` — exhaustive small-state sum, not GPU-shaped |
| **Workload benchmarks** | ✅ 2 composite chains, **all driven through public APIs**, every native row carries a `dispatched_on_gpu` proof bit | `ga_feature_pipeline` (decisive native win); `ebt_tiny_refinement` (loses at tiny default shape — honest reporting) |
| **EBT-tiny break-even sweep** | ✅ opt-in mode (`--ebt-sweep`), summary tags each shape with `status="native_dispatched"` or `"degraded_fallback"` | Widened **streaming closed-form** kernel (any `D`; `K ≤ 256`). Recent M-series run: first native win at `B=16,K=32,D=128/T=8` (~1.1×); peak **~55× at `B=64,K=128,D=1024/T=256`**. Numbers will drift across hosts — the proof bit is the stable contract. |
| **GA / EBM via `tessera.ga.*` / `tessera.ebm.*`** | 🟢 **integration gap fully closed** | **17 / 17 GA + 9 / 9 native EBM** ops route through [`tessera._apple_gpu_dispatch`](../../python/tessera/_apple_gpu_dispatch.py) (incl. `ebm.ebt_tiny`) |
| **JIT / compiler bridge** | ✅ **landed for all 26 fast paths** | [`tessera.compiler.jit_bridge`](../../python/tessera/compiler/jit_bridge.py) — Python frontend → manifest resolve → shared loader dispatch + thread-local route trace. **All 17 GA + 9 EBM** fast paths call `dispatch_via_manifest`; every public-API GPU dispatch produces a `JitBridgeRoute` row that records `(op, target, status, symbol, context, latency_ms)`. The benchmark's native EBM primitive rows + the JIT-bridge benchmark rows + the workload rows all use the trace as their proof-of-dispatch bit. |
| **Compiler vertical slice** | ✅ **landed** | [`tessera.compiler.clifford_jit`](../../python/tessera/compiler/clifford_jit.py) — `@clifford_jit(target="apple_gpu")` decorator traces a Clifford-only function on first call, captures the op plan from the bridge trace, verifies every op against the manifest, and freezes a `CliffordCompiledArtifact` carrying the plan + plan hash + Apple target metadata.  Benchmark `vertical_slice` row embeds the full artifact JSON so reports show the compile-time plan. v1 demo: `point_cloud_rotor_invariant` (`rotor_sandwich → norm`). |
| **Fused GA + EBM workload** | ✅ **landed** | `rotor_conditioned_ebt` — `ga.exp_mv → ga.rotor_sandwich → ebm.ebt_tiny` through public APIs, all bridge-traced, native ~20× speedup vs the equivalent numpy chain on a recent M-series run. |
| **Metal buffer pool** | 🟡 **partial — heavy-use dispatchers migrated** | `MetalDeviceContext` keeps a 19-bucket shared-storage buffer pool keyed by size class.  `dispatch_clifford_unary_8x8_f32_msl` and `dispatch_ebm_ebt_tiny_refinement_argmin_f32_msl` recycle buffers via `metal_buffer_acquire` / `metal_buffer_release`; the remaining ~25 dispatchers still call `newBufferWithBytes` directly — migration is mechanical, open follow-up. |
| **Build / test gate** | ✅ deterministic CI test, **in `scripts/validate.sh` spine** | [`tests/unit/test_benchmark_ga_ebm.py`](../../tests/unit/test_benchmark_ga_ebm.py) — 118 tests + 20-test [`tests/unit/test_jit_bridge.py`](../../tests/unit/test_jit_bridge.py), graceful non-Darwin skip |

## What's claimed

- **17 / 17 GA primitives** ship fused MSL kernels on Apple GPU, each end-to-end benchmarked (Python API → manifest lookup → ctypes dispatch → Metal execution → correctness check vs Python reference). Manifest source of truth: [`backend_manifest.py::_CLIFFORD_APPLE_GPU_FUSED`](../../python/tessera/compiler/backend_manifest.py).
- **8 / 9 EBM primitives** ship fused MSL kernels on Apple GPU:
  `ebm_inner_step`, `ebm_refinement`, `ebm_langevin_step`,
  `ebm_decode_init`, `ebm_bivector_langevin` (kernel-reuse — same MSL
  symbol as `ebm_langevin_step` on grade-projected inputs),
  `ebm_sphere_langevin`, `ebm_self_verify`, and `ebm_energy` (quadratic
  specialization).  Manifest: [`_EBM_APPLE_GPU_FUSED`](../../python/tessera/compiler/backend_manifest.py).
- **Apple GPU C ABI surface**: GA primitive symbols + 7 EBM symbols (one EBM
  kernel is reused for `bivector_langevin`), all in
  [`apple_gpu_runtime.mm`](../../src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm).
- **Workload mode**: two composite benchmark chains stringing primitives together (`ga_feature_pipeline` + `ebt_tiny_refinement`), each emitting `apple_gpu` + `python_ref` rows so speedup is a single subtraction.
- **EBT-tiny break-even sweep**: opt-in `--ebt-sweep` flag emits one apple_gpu + python_ref row per `(B, K, D, T)` point. After the streaming closed-form kernel rewrite, every sweep row carries a `dispatched_on_gpu` bit + a per-shape `status` (`native_dispatched` vs `degraded_fallback`); the summary only computes `speedup` for shapes that actually fired on-device. Recent M-series run: first native win at `B=16,K=32,D=128/T=8` (~1.1×); peak ~55× at `B=64,K=128,D=1024/T=256`. Numbers drift across hosts; the proof bits + status fields are the stable contract.
- **`tessera.ga.inner` and `tessera.ebm.inner_step` dispatch through `tessera._apple_gpu_dispatch`** on Apple Silicon — no benchmark-local ctypes required to get the GPU speedup. The dispatcher caches the compiled dylib + bound symbols across calls.
- **JIT / compiler bridge ([`tessera.compiler.jit_bridge`](../../python/tessera/compiler/jit_bridge.py))** wires the four-stage pipeline: (1) the Python frontend calls `dispatch_via_manifest(op_name, args)`, (2) the bridge resolves the apple_gpu C ABI symbol via `manifest_for(op_name)`, (3) it binds the symbol through the shared loader, (4) it records a `JitBridgeRoute(op, target, status, symbol, context, latency_ms)` row in the thread-local trace. Inside a `jit_context("apple_gpu")` span the trace marks the context as `jit:apple_gpu`. The benchmark exposes this via `namespace="jit_bridge"` rows with a `routes` column proving each dispatch went through the bridge end-to-end.
- **CI health check**: `python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci` returns 0 + a parseable JSON report when Apple Silicon + clang++ are present; exits cleanly with `skipped_apple_gpu` reason on non-Darwin.  Wired into [`scripts/validate.sh`](../../scripts/validate.sh) so the full validation spine runs it.

## Tested hardware / toolchain

- **Apple Silicon M-series** (laptop class), macOS 15+, Xcode 16+ command-line tools.
- `clang++` 19 from the Apple toolchain; Metal / MetalPerformanceShaders / Foundation frameworks.
- Python 3.14 + numpy.
- The runtime dylib is compiled on demand by the benchmark / test harness — no pre-build required.

## Known non-claims (do not over-promise)

1. **EBT-tiny native wins are real, but the precise numbers drift.**
   The 2026-05-17 streaming closed-form kernel rewrite removed the
   per-thread D-vector register buffer (which had a hard 256-element
   cap) — D is now unbounded; K is still bounded at 256 by the
   threadgroup-size budget for the K-way argmin reduction.  Every
   native row in the report carries a `dispatched_on_gpu` proof bit:
   - **Native EBM primitive rows + JIT-bridge benchmark rows** —
     proof bit sourced from the `tessera.compiler.jit_bridge` route
     trace (a one-shot trace span around the timed dispatch).
   - **EBT-tiny workload + `--ebt-sweep` rows** — proof bit sourced
     from `tessera.ebm.ebt_tiny_dispatched_on_gpu()`; the sweep
     summary additionally tags each shape with
     `status="native_dispatched"` vs `"degraded_fallback"`.

   Silent numpy fallbacks (e.g., `K > 256`) degrade the row's
   `backend` to `python_ref`, its `ok` field to `False`, and (for
   sweep rows) flag the shape as `degraded_fallback` instead of
   reporting it as a native win.  Headline speedups (~55× peak at
   `B=64,K=128,D=1024,T=256` on a recent M-series run) will drift
   across hosts and toolchain versions; the schema + proof bits are
   the stable contract.

2. **Some single-op EBM rows have native-vs-Python latency that is
   *worse* at small shapes.** Single-element pointwise kernels
   (`ebm_inner_step` at n=64 floats) are dominated by Metal dispatch
   overhead (~0.2 ms); numpy does the same work in ~1 µs.  The
   `ga_feature_pipeline` workload still wins 13× because the chained
   primitives amortize dispatch over real arithmetic.

3. **Public-API GPU coverage is complete** for both GA (17/17) and
   native EBM (9/9 including the `ebm.ebt_tiny` fused pipeline).
   Every `tessera.ga.*` and `tessera.ebm.*` user call that has a
   fused MSL kernel transparently routes through
   `tessera._apple_gpu_dispatch` when the input shape + dtype match
   the manifest contract; on non-Darwin or with a mismatched dtype
   the numpy reference path runs instead.
   **JIT-bridge coverage is also complete** (26 of 26 fast paths):
   all 17 GA ops + all 9 native-EBM ops call
   `jit_bridge.dispatch_via_manifest`, so every public-API GPU
   dispatch produces a `JitBridgeRoute` row when tracing is on.
   The benchmark uses the trace as proof of dispatch for native EBM
   primitive rows + JIT-bridge benchmark rows.

4. **No on-device RNG yet.**  The native `langevin_step` /
   `decode_init` / `sphere_langevin` kernels take a *host-supplied
   noise buffer* (deterministic from a `tessera.rng.RNGKey`).
   On-device Philox is a follow-up sprint.

5. **EBT refinement uses a fixed gradient snapshot.**  The native
   `ebm_refinement` kernel runs T iterations of `y - eta*grad` with
   the same `grad` buffer reused at every step.  Real EBT recomputes
   `grad = dE/dy` per step; that needs the energy_fn lifted to MSL
   (related to native `ebm_energy` work — the v1 `ebm_energy`
   specialization only covers the quadratic case).

6. **NVIDIA / AMD / Cerebras / Metalium**: GA + EBM manifest entries
   for these targets read `status="planned"` — gated on Phase G / H / I.

## How to run

```bash
# CI health check — what `tests/unit/test_benchmark_ga_ebm.py` runs.
python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci

# Full sweep (50 reps default, stable percentiles):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --output /tmp/ga_ebm.json

# Workload-only (skip per-primitive rows):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --workloads-only

# Tune the EBT refinement depth (default 8 inner steps):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --refinement-T 32

# Sample report (checked in for schema reference, latencies illustrative):
benchmarks/apple_gpu/sample_ga_ebm_report.json
```

## Next targets

In priority order (descending):

1. **Finish the buffer-pool sweep.**  Two of ~27 dispatchers
   currently recycle via `metal_buffer_acquire` / `_release`; the
   remaining ones still call `newBufferWithBytes` directly.
   Migration is mechanical (one diff per helper) and brings every
   small-shape kernel's dispatch floor down uniformly.  Pair with
   async command-queue scheduling (don't block `waitUntilCompleted`
   inside the helper) for the next big perf step.

2. **On-device RNG (Philox in MSL).**  The native `langevin_step` /
   `decode_init` / `sphere_langevin` kernels take a host-supplied
   noise buffer today; Philox-in-MSL lets the kernels generate
   their own deterministic noise from a 4-element seed.  Unblocks
   T-step Langevin chains where re-uploading noise each step is
   the dominant cost.

3. **Lift arbitrary `energy_fn` to MSL** (paired with #2).  v1
   `ebm_energy` is a quadratic specialization; a restricted
   Python-AST-to-MSL visitor (handles polynomial + a handful of
   activations) would enable real EBT refinement with per-step
   gradient recomputation natively, instead of fixed-gradient
   snapshots.

4. **Broaden `@clifford_jit`** beyond the v1 Cl(3,0) f32 surface:
   add Cl(1,3) support, fp16 dtype, and (longer term) lift the
   "trace once at first call" model to a real AST → IR lowering so
   shape-dependent branches don't silently fall off the plan.

5. **More fused workloads** alongside `rotor_conditioned_ebt` and
   `ebt_tiny`.  Candidates: a sphere-Langevin chain with
   on-device retraction loop, a GA-conditioned diffusion step
   (`rotor_sandwich` + `langevin_step` chained).  Same pattern:
   identify the multi-dispatch composition that loses to numpy at
   small shapes and write one MSL kernel that consumes the whole
   chain.

6. **`ebm_partition_exact` stays reference** — small-state
   exhaustive sums are not GPU-shaped at typical scale.

7. **NVIDIA / AMD / Cerebras / Metalium** GA + EBM coverage — gated
   on Phase G / H / I.

## Sources

- Manifest: [`python/tessera/compiler/backend_manifest.py`](../../python/tessera/compiler/backend_manifest.py)
- Runtime: [`src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm`](../../src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm)
- Benchmark: [`benchmarks/apple_gpu/benchmark_ga_ebm.py`](../../benchmarks/apple_gpu/benchmark_ga_ebm.py)
- Benchmark README: [`benchmarks/apple_gpu/README.md`](../../benchmarks/apple_gpu/README.md)
- Roadmap (full history): [`docs/audit/ga_ebm_roadmap.md`](../audit/ga_ebm_roadmap.md)
- Earlier gap audit: [`docs/audit/apple_ga_ebm_native_execution_gap.md`](../audit/apple_ga_ebm_native_execution_gap.md)
- CI test: [`tests/unit/test_benchmark_ga_ebm.py`](../../tests/unit/test_benchmark_ga_ebm.py)
