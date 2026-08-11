---
last_updated: 2026-08-10
audit_role: reference
owning_plan_item: async-contract reconciliation follow-ups
---

# Strix Halo (AMD box) worklist — async-contract session follow-ups

Items surfaced during the 2026-08-10 `tile.async_copy`/`tile.wait_async`
contract reconciliation (branch `claude/happy-jennings-a07443`; see the dated
entry at the top of [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md)) that cannot be
closed on the Mac and belong on the primary box (fleet routing:
`INTEGRATED_COMPILER_PLAN.md` §6a). Ordered by risk to the landed change.

## 1. ROCm lit lane over the reconciled branch — the one real gate left

The Mac build (CPU + Apple) produces no `tessera-rocm-opt`, so neither
`check-tessera-rocm` nor the 29 ROCm/x86-lane fixtures under
`tests/tessera-ir/` ran against this change. The legacy attr envelope the new
contract declares (`tile.barrier_id` / `tile.depends_on`) is load-bearing
exactly on this box's lanes (`TileToROCM.cpp`, `ROCMWaveLdsPipeline.cpp`,
`GenerateWMMAGemmKernel.cpp`), and the new `TILE_ASYNC_STAGE_NEGATIVE` check
now runs on every `tile.async_copy` those pipelines emit.

```bash
ninja -C build            # ALL targets — single-target builds hide link gaps
ninja -C build check-tessera-ir
ninja -C build check-tessera-rocm
```

Expected: green. No ROCm producer emits a negative `stage` (audited: stage
attrs are only attached to `tile.pipeline_init`, always 0), so a red here
means an unaudited producer — investigate before relaxing anything.

## 2. gfx1151 hardware suites that ride the async-copy seam

`tests/unit/test_rocm_ssa_lds_pipeline_benchmark.py` builds and executes the
wave-LDS pipeline whose copies/waits the reconciled verifiers now inspect.
Hardware-executing on this box only.

```bash
source .venv/bin/activate && export PYTHONPATH=python
python3 -m pytest tests/unit/test_rocm_ssa_lds_pipeline_benchmark.py -v
```

## 3. Cross-check the Mac pytest failures against this environment

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

## 4. `examples/optimization` x86-intrinsics targets break `ninja -C build` on ARM

`01_loop_tiling_blocking` and `02_vectorization_intrinsics` include
`<immintrin.h>` unconditionally, so a full Mac build always fails at these two
targets — which this session showed is not cosmetic: ninja stopping there left
`tessera-opt` stale and produced 6 phantom lit failures until relinked. Fix:
gate the two example targets on `CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"`
in `examples/optimization/CMakeLists.txt`. The fix is writable anywhere; the
"x86 still builds them" verification belongs here.

## 5. Confirm main's lit health after the PR #543/#544 fixture fix

**Resolved 2026-08-10 by PR #544 — re-confirm here only.**
`phase2/effect_annotation.mlir` was red on main because PR #543's
`@canonical_rng` used `"tessera.rng_uniform"`, an op no ODS declared, and the
tessera Graph dialect rejects unknown ops. PR #544 landed
`Tessera_RNGUniformOp` / `Tessera_RNGNormalOp` in `TesseraOps.td` with real
verifiers plus a negative fixture (`phase2/rng_stateful_invalid.mlir`); both
pass on the Mac. Re-run `check-tessera-ir` here so the full-build config
confirms it — a Mac pass alone leaves the ROCm-registered parse path
unchecked.

## Explicitly NOT this box's work

- The unbuilt-dead-code audit (`ScheduleOps.cpp` remainder, `src/compiler/mlir/`)
  — build-system archaeology, Mac-doable, session already running.
- Apple-lane anything (Decision: Mac-only).
