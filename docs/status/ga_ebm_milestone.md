# GA / EBM milestone status

> **One-page canonical status.** Update this page when something
> changes; everything else in the repo (READMEs, roadmap, audit) cites
> the *claims* below rather than restating them.
>
> **Last updated:** 2026-05-17 (GA11 + EBM-broadening + workload mode + **fused refinement kernel + 7 public-API GPU paths**).

## TL;DR

| Surface | Status | Coverage |
|---|---|---|
| **GA primitives (Clifford)** | ✅ all native + benchmarked | 17 / 17 fused MSL kernels on Apple GPU; end-to-end report rows for every primitive |
| **EBM primitives (native)** | 🟡 close to complete, 1 to go | **8 / 9** with fused MSL kernels (`ebm_refinement` now fused in a single dispatch — was T-step ping-pong) |
| **EBM primitives (Python ref only)** | ⏳ 1 op | Only `ebm_partition_exact` — exhaustive small-state sum, not GPU-shaped |
| **Workload benchmarks** | ✅ 2 composite chains, **driven by public APIs** | `ga_feature_pipeline` (**27× speedup** vs Python ref at B=32); `ebt_tiny_refinement` (K-cand × T-step loop) |
| **EBT-tiny break-even sweep** | ✅ opt-in mode (`--ebt-sweep`) | 7-point (B, K, D, T) ladder; **first native win at `B=32,K=128,D=512/T=64` (6.5×); peak 20.3× at `B=64,K=128,D=1024/T=256`** |
| **GA / EBM via `tessera.ga.*` / `tessera.ebm.*`** | 🟢 **integration gap closed for 8 ops** | `ga.inner`, `ga.exp_mv`, `ga.rotor_sandwich`, `ga.norm`, `ebm.inner_step`, `ebm.refinement`, `ebm.self_verify`, `ebm.energy_quadratic`, `ebm.decode_init` all route through [`tessera._apple_gpu_dispatch`](../../python/tessera/_apple_gpu_dispatch.py) |
| **Build / test gate** | ✅ deterministic CI test, **in `scripts/validate.sh` spine** | [`tests/unit/test_benchmark_ga_ebm.py`](../../tests/unit/test_benchmark_ga_ebm.py) — 96 tests, graceful non-Darwin skip |

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
- **EBT-tiny break-even sweep**: opt-in `--ebt-sweep` flag emits one apple_gpu + python_ref row per `(B, K, D, T)` point and summarizes `first_native_win_shape` in the envelope. At the v1 sweep ladder the native row hasn't beaten numpy yet — see "Known non-claims" #1 for why.
- **`tessera.ga.inner` and `tessera.ebm.inner_step` dispatch through `tessera._apple_gpu_dispatch`** on Apple Silicon — no benchmark-local ctypes required to get the GPU speedup. The dispatcher caches the compiled dylib + bound symbols across calls.
- **CI health check**: `python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci` returns 0 + a parseable JSON report when Apple Silicon + clang++ are present; exits cleanly with `skipped_apple_gpu` reason on non-Darwin.  Wired into [`scripts/validate.sh`](../../scripts/validate.sh) so the full validation spine runs it.

## Tested hardware / toolchain

- **Apple Silicon M-series** (laptop class), macOS 15+, Xcode 16+ command-line tools.
- `clang++` 19 from the Apple toolchain; Metal / MetalPerformanceShaders / Foundation frameworks.
- Python 3.14 + numpy.
- The runtime dylib is compiled on demand by the benchmark / test harness — no pre-build required.

## Known non-claims (do not over-promise)

1. **`ebm_refinement` now wins at the right scale** (was open in the
   previous revision of this doc). The new fused MSL kernel runs all
   T inner-step iterations inside a single dispatch — each thread
   keeps its `y_i` in a register and loops T times.  Break-even at
   `(B=32, K=128, D=512, T=64)`; peak speedup 20.3× at
   `(B=64, K=128, D=1024, T=256)`.  At sub-millisecond numpy times
   (small shapes) the native kernel still loses to dispatch overhead
   — that's reported honestly via the sweep summary.

2. **Some single-op EBM rows have native-vs-Python latency that is
   *worse* at small shapes.** Single-element pointwise kernels
   (`ebm_inner_step` at n=64 floats) are dominated by Metal dispatch
   overhead (~0.2 ms); numpy does the same work in ~1 µs.  The
   `ga_feature_pipeline` workload still wins 13× because the chained
   primitives amortize dispatch over real arithmetic.

3. **Most GA / EBM Python API calls still go through numpy.** As of
   this update, **9 ops** route through the dispatcher transparently:
   `ga.inner` / `ga.exp_mv` / `ga.rotor_sandwich` / `ga.norm` and
   `ebm.inner_step` / `ebm.refinement` / `ebm.self_verify` /
   `ebm.energy_quadratic` / `ebm.decode_init` (the last only when a
   `mean` kwarg is provided).  The remaining 13 GA primitives + 4
   native-EBM primitives reach the GPU today only via the benchmark
   or by importing the runtime symbols directly.  Sweep-through is
   open work — see "Next targets".

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

1. **Continue the public-API sweep**.  9 ops route through
   `_apple_gpu_dispatch` today; the remaining 13 GA primitives
   (reverse / grade_involution / conjugate / hodge_star / log_mv /
   geometric_product / wedge / left_contraction / grade_projection /
   ext_deriv / vec_deriv / codiff / integral) + 4 native-EBM
   primitives (langevin_step / bivector_langevin / sphere_langevin /
   energy with arbitrary energy_fn) all need the same `_try_apple_gpu_*`
   helper + Python frontend hookup.  The pattern is small + repeatable.

2. **Reduce per-dispatch host overhead** so small-shape EBM rows stop
   losing to numpy.  Each `_try_apple_gpu_*` helper today allocates a
   fresh output buffer + does a `MTLBuffer newBufferWithBytes` copy.
   A per-thread or per-process buffer pool + `MTLResourceStorageModeShared`
   with direct pointers could drop the floor latency from ~0.2 ms to
   ~50 µs and let the workload break-even drop dramatically.

3. **On-device RNG (Philox in MSL)** for `langevin_step` /
   `decode_init` / `sphere_langevin` — removes the host-side noise
   pre-generation step + makes the kernels self-contained.

4. **Lift arbitrary `energy_fn` to MSL** so the v2 `ebm_energy`
   kernel isn't just a quadratic specialization.  Likely path:
   restricted Python AST → MSL via a small visitor (covers
   polynomials + a handful of activations).  Enables real EBT
   refinement with per-step gradient recomputation natively.

5. **`ebm_partition_exact` stays reference** — small-state exhaustive
   sums are not GPU-shaped at typical scale.

6. **NVIDIA / AMD / Cerebras / Metalium** GA + EBM coverage — gated
   on Phase G / H / I.

## Sources

- Manifest: [`python/tessera/compiler/backend_manifest.py`](../../python/tessera/compiler/backend_manifest.py)
- Runtime: [`src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm`](../../src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm)
- Benchmark: [`benchmarks/apple_gpu/benchmark_ga_ebm.py`](../../benchmarks/apple_gpu/benchmark_ga_ebm.py)
- Benchmark README: [`benchmarks/apple_gpu/README.md`](../../benchmarks/apple_gpu/README.md)
- Roadmap (full history): [`docs/audit/ga_ebm_roadmap.md`](../audit/ga_ebm_roadmap.md)
- Earlier gap audit: [`docs/audit/apple_ga_ebm_native_execution_gap.md`](../audit/apple_ga_ebm_native_execution_gap.md)
- CI test: [`tests/unit/test_benchmark_ga_ebm.py`](../../tests/unit/test_benchmark_ga_ebm.py)
