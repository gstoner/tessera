# Saved LSE and sparse Schedule handoff

Owner E2E-REAL-6 / ROCM-2; sync GFX1201-SAVED-SPARSE-SCHEDULE-2026-09-13.
Measured on Tajasarus gfx1201, ROCm 10.0 / HIP 7.15, Ubuntu WSL2,
LLVM 23.1.1 assertions enabled. Sources remain uncommitted and are bound by
`source-identity.json`.

## Proven envelope

- Explicit `saved` attention backward packages retain forward O/LSE in private
  resident storage. Three different cotangents use one forward launch, preserve
  the captured Q/K/V/bias despite caller mutation, reuse allocations, and match
  the reference for fp16 and bf16. `auto` remains recompute, with no borrowed
  gfx1151 threshold or performance promotion.
- Same-device external gradient readers delay saved-frame retirement. This is
  first-order backward ownership, not exported LSE pointers, arbitrary public
  tape composition or recoverable uncertain driver teardown.
- `sparse-schedule.json` and the profiled repetition exercise registered
  `schedule.sparse_mma` → `tile.sparse_mma` → `tessera_rocm.swmmac` → LLVM/HSACO.
  Six exact f16/bf16 comparisons cover all 2:4 index pairs. Source and image hashes
  are recorded. Verifier tests refuse gfx1151 at Schedule and Tile boundaries.
  The packing API owns immutable register bytes; no input pruning occurs.

## Still open

The sparse stages are internal physical fragment IR. Public logical sparse Graph
capture, automatic packing in generated code, general matrix tiling and production
package/runtime binding remain open. A probe that consumes packed memrefs is not
closure of those public interfaces. No new public op/dtype/AD rule is claimed.

`profiler-counts.json` records 98 API regions but zero kernel dispatch,
code-object, kernel-symbol and counter-event rows. `/dev/kfd` is absent. Hence
clock calibration and kernel attribution remain blocked on this WSL setup.
Positive HIP event durations are now only `device_event_samples_valid`;
`device_event_selector_eligible` stays false. No performance promotion.

## Reproduce on the owning host

```sh
source ~/.config/tessera/env.sh
TESSERA_GFX1201_DEVICE_PROOF=1 python -m pytest -q tests/unit/test_rocm_gfx1201_scheduled.py tests/unit/test_resident_rocm_attention.py tests/unit/test_rocm_sparse_packing.py
python -m benchmarks.rocm.record_gfx1201_sparse_wmma --route schedule_mlir --output /tmp/sparse.json
rocprofv3 --hip-trace --kernel-trace -f rocpd -d /tmp/sparse-profile -- python -m benchmarks.rocm.record_gfx1201_sparse_wmma --route schedule_mlir --output /tmp/sparse-profiled.json
```

No Apple, NVIDIA, gfx1151 or x86 execution evidence transfers from this packet.
