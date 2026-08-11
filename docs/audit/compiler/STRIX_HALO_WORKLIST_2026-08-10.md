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

The full `-m "not slow"` sweep on the Mac (rebased tree, post
`TesseraAppleRuntimeShared` rebuild) ended **16 failed / 13686 passed /
3478 skipped**. The two identified so far are
`test_solver_ift_artifact.py::test_solver_ift_tile_artifacts_reach_architecture_owned_lowering`
and
`test_stdlib_dspark_perf.py::test_ds2_runtime_launch_overhead_is_bounded_against_ds1_oracle`
— neither touches the async contract, but the triage is incomplete (full list
pending; update this section when it lands). Run the same sweep here to
separate environment-dependent failures (Metal perf bounds, Mac-skipped
x86/ROCm lanes) from real regressions on main.

## 4. `examples/optimization` x86-intrinsics targets break `ninja -C build` on ARM

`01_loop_tiling_blocking` and `02_vectorization_intrinsics` include
`<immintrin.h>` unconditionally, so a full Mac build always fails at these two
targets — which this session showed is not cosmetic: ninja stopping there left
`tessera-opt` stale and produced 6 phantom lit failures until relinked. Fix:
gate the two example targets on `CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"`
in `examples/optimization/CMakeLists.txt`. The fix is writable anywhere; the
"x86 still builds them" verification belongs here.

## 5. Confirm main's lit health after the PR #543 fixture fix

`phase2/effect_annotation.mlir` is red on main itself: PR #543's
`@canonical_rng` uses `"tessera.rng_uniform"`, an op no ODS declares, and the
tessera Graph dialect rejects unknown ops. A fix session was spawned
(2026-08-10); once it lands, re-run `check-tessera-ir` here so the full-build
config confirms it — a Mac pass alone leaves the ROCm-registered parse path
unchecked.

## Explicitly NOT this box's work

- The unbuilt-dead-code audit (`ScheduleOps.cpp` remainder, `src/compiler/mlir/`)
  — build-system archaeology, Mac-doable, session already running.
- Apple-lane anything (Decision: Mac-only).
