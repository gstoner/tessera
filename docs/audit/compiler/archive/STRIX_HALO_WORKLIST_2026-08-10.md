---
last_updated: 2026-08-10
audit_role: plan
plan_state: closed
owning_plan_item: TILE-SYNC-RECONCILE-2026-08-10
---

# Strix Halo (AMD box) worklist — async-contract session follow-ups

Items surfaced during the 2026-08-10 `tile.async_copy`/`tile.wait_async`
contract reconciliation, subsequently merged by PR #544. This plan separates
the compiler-contract gates that PR #544 closed from exact-device and
host-specific follow-ups. Host-free CI evidence never substitutes for gfx1151,
Zen 5, or Mac evidence (fleet routing: `INTEGRATED_COMPILER_PLAN.md` §6a).

## Closed by PR #544

- **ROCm compiler parity:** the required `Validate / rocm compiler (host-free
  LLVM/MLIR 23)` lane passed on PR #544. This closes parse, registration,
  verifier, and host-free lowering parity for the reconciled contract. It does
  not execute a gfx1151 kernel.
- **Core build and unit parity:** the required build, unit, lint, audit, and
  fan-in lanes passed on PR #544. The x86 fixtures sharing the core Tile
  dialect therefore no longer need a generic "primary-box full build" gate;
  exact AVX-512 behavior remains architecture-owned.
- **Graph RNG fixture:** `tessera.rng_uniform` and `tessera.rng_normal` are now
  registered ODS operations with verifiers and random-effect annotation, so
  the PR #543 `phase2/effect_annotation.mlir` parse failure is closed.
- **Dead build dependency:** the deleted queue TableGen target was removed
  from `TESSERA_COMPILER_FOUNDATION_DEPS`, restoring the aggregate build.

This scoped host worklist owns only the acceptance checks below. The compiler
map [`README.md`](README.md) and global sequencing authority
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) determine when
follow-on compiler work is selected.

## Closed 1 — compiler benchmark plus exact gfx1151 async-copy/LDS execution

The host-free lane cannot prove that the compatibility attributes consumed by
`TileToROCM.cpp`, `ROCMWaveLdsPipeline.cpp`, and
`GenerateWMMAGemmKernel.cpp` preserve execution. Run the wave-LDS suite on the
gfx1151 host and retain its architecture, toolchain, artifact digest,
correctness, resource, and timing provenance. Do not promote from WSL
synchronized-host timing alone.

```bash
source .venv/bin/activate && export PYTHONPATH=python
# Structural unit contract only; this does not launch a GPU.
python3 -m pytest tests/unit/test_rocm_ssa_lds_pipeline_benchmark.py -v

# Repeated host-compiler planning/lowering measurements. The emitted schema is
# deliberately evidence=host_compiler_only and device_latency_ms=null.
TESSERA_OPT=build-rocm-7.14-llvm23-clean/tools/tessera-opt/tessera-opt \
  python3 benchmarks/rocm/benchmark_rocm_ssa_lds_pipeline.py

# Actual gfx1151 execution across the affected async-copy/LDS seam.
TESSERA_BUILD_DIR=build-rocm-7.14-llvm23-clean \
  python3 -m pytest \
    tests/unit/test_rocm_async_copy_runnable.py \
    tests/unit/test_rocm_wmma_runtime_symbol.py \
    tests/unit/test_rocm_pipeline_tile_lowering.py \
    -k 'global_to_lds or wmma_lds or wmma_pipe or via_tile_matches' -v
```

The earlier form of this worklist incorrectly described
`test_rocm_ssa_lds_pipeline_benchmark.py` as hardware-executing. Its source and
benchmark schema explicitly prove only structural and host-compiler behavior;
the three-test command above is the exact-device gate.

Validated on the WSL-visible Radeon 8060S/gfx1151 with ROCm 7.14 and LLVM/MLIR
23 on 2026-08-10:

- ROCm lit: **56/56 passed**;
- shared Tessera IR lit: **302 passed, 52 configuration-unsupported**;
- focused structural/device cohort: **16/16 passed**, including the
  global→LDS round trip, five LDS-staged WMMA shapes, five two-stage pipelined
  WMMA shapes, and the bit-identical via-Tile/production comparison;
- seven-run host-compiler medians: planner **15.1811–15.4441 ms** and complete
  lowering **16.5207–16.7524 ms** over one/two/four/eight stages, with zero
  legacy buffer references and no surviving portable allocations/pipeline
  operations.

These timing values are `host_compiler_only`; `device_latency_ms` is null. The
run closes correctness for this reconciliation but makes no selector or
device-performance promotion.

## Closed 2 — confirm the attributed Mac failures on the owning WSL lanes

**Triage completed 2026-08-10 (PR #544 integration run) — no longer a
blocker, but worth a confirming sweep here.** The full `-m "not slow"` sweep on
the Mac over the integrated tree ended **16 failed / 14064 passed / 3071
skipped**, and every failure was attributed:

- **11 reproduce identically with `origin/main`'s `python/tessera` + `tests/unit`
  checked out**, so they are Mac host-state, not regressions:
  `test_apple_gpu_spectral`, `test_apple_gpu_delta_erase_routing`,
  `test_apple_legacy_retune_benchmark` (Apple-GPU numerics / live-host ledger),
  the four `test_lit_env_overrides` cases, `test_rocm_pipeline_tile_lowering`
  (×3), `test_scheduled_attention_backward_consumers`, and
  `test_solver_ift_artifact` — the last five being exactly the gfx1151/ROCm
  lanes this box owns.
- **5 are load-induced timing flakes**, all `*_perf_baseline_is_bounded`-style
  wall-clock bounds (`test_rocm_{dequant_gemm,mla_decode_step,moe_transport,
  sparse_attn}_compiled`, `test_stdlib_dspark_perf::test_ds2_*`). They fail
  under full-suite parallel load and **pass on re-run in isolation**.

What this box adds: the 5 ROCm/gfx1151 entries in the first bucket are
Mac-environmental *there* but should be **green here**. If any of them is red
on this box, it is a real regression — that is the signal to look for, not the
raw failure count.

The owning WSL rerun passed **25/25** tests across
`test_rocm_pipeline_tile_lowering.py`,
`test_scheduled_attention_backward_consumers.py`, and
`test_solver_ift_artifact.py`, including the five cases attributed to the Mac
environment. This closes the cross-host confirmation without changing any
performance selector.

## Closed 3 — make `examples/optimization` portable at configure time

`01_loop_tiling_blocking` and `02_vectorization_intrinsics` include
`<immintrin.h>` unconditionally, so a full Mac build always fails at these two
targets — which this session showed is not cosmetic: ninja stopping there left
`tessera-opt` stale and produced 6 phantom lit failures until relinked. Fix:
gate the two example targets on `CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"`
in `examples/optimization/CMakeLists.txt`. The fix is writable anywhere; the
"x86 still builds them" verification belongs on an x86 build. This is a source
portability fix, not an AVX-512 performance or exact-device claim.

PR #546 landed the processor guard. Both intrinsic targets rebuild on the x86
WSL host, and an `aarch64` CMake toolchain configuration excludes them while
building the two portable examples.

## Closed 4 — confirm main's lit health after the PR #543/#544 fixture fix

**Resolved 2026-08-10 by PR #544 — re-confirm here only.**
`phase2/effect_annotation.mlir` was red on main because PR #543's
`@canonical_rng` used `"tessera.rng_uniform"`, an op no ODS declared, and the
tessera Graph dialect rejects unknown ops. PR #544 landed
`Tessera_RNGUniformOp` / `Tessera_RNGNormalOp` in `TesseraOps.td` with real
verifiers plus a negative fixture (`phase2/rng_stateful_invalid.mlir`). The
refreshed WSL compiler build passes ROCm lit **56/56** and shared Tessera IR lit
**302 passed / 52 configuration-unsupported**, including
`phase2/effect_annotation.mlir`, `phase2/rng_stateful_invalid.mlir`, and
`phase2/pm_verify_async_token.mlir`.

## Exit criteria

Archive this plan only after:

1. the gfx1151 SSA/LDS suite passes with explicit structural/device evidence
   (**complete 2026-08-10**);
2. the five ROCm/gfx1151 failures attributed to the Mac environment pass on
   this WSL host (**complete 2026-08-10; 25/25 containing cohort**); and
3. the optimization examples configure correctly for ARM and build on x86
   (**complete 2026-08-10**).

All exit criteria are complete. This plan is archived; subsequent async-token
carrier and overlap-scheduling work is owned by `COMP-SCHED-OVERLAP-1` in the
integrated compiler plan.

## Explicitly NOT this box's work

- The dead `ScheduleOps.cpp` and `src/compiler/mlir/` plugin surfaces — deleted
  by PR #544.
- Apple-lane anything (Decision: Mac-only).
