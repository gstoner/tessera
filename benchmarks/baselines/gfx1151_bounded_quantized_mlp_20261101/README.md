# Bounded-row native quantized MLP on gfx1151 (2026-11-01)

**Sync key:** `GFX1151-NVFP4-BOUNDED-MLP-20261101` (program `GFX1151-QUANTIZED-MLP-TIMING-20261101`).
**Owner:** ROCM-NVFP4-INGEST-1. The gfx1151 counterpart of
`../gfx1201_bounded_quantized_mlp_20261027/`. Correctness for the v5 program on
gfx1151 is slice 4 (PR #1029). This packet adds timing only.

**Host:** Princess-Luna, Radeon 8060S (gfx1151, RDNA 3.5), Ubuntu 26.04 under
WSL2, ROCm 10.0 / HIP 7.15.26333, apt LLVM/MLIR 23.1.1 (assertions OFF) `build/`
tree. Ran unsandboxed under `flock /tmp/tessera-gfx1151-device.lock`, and the
recorder checked the lock was held (`device_lock.held: true`).

`paired.json` was produced by `benchmarks/rocm/diagnose_bounded_quantized_mlp.py`
at commit `070f445d7` on a clean worktree (`worktree_dirty: false`). It records
the recorder, compiler and provider hashes, `route`, `latency_source` and
`timing_source`. A never-timed warm-up owner was prepared first
(`warm_up_owner: true`).

## Method

One bounded owner was prepared at capacity 256, with H/K/N = 128/12288/256.
Each static owner was compiled and prepared at its own M. The arms alternated for
12 rounds. Each value is the owner's HIP-event window (`hip_event_owner_window`)
averaged over 50 repeats; the table shows the median. The activation update is
host wall time (`host_wall_perf_counter`), a separate domain. Outputs were
compared bitwise before timing.

## Result

| Active M | Bounded µs | Static µs | Bounded / static | Bitwise equal | Activation update ms | Static compile s |
|---|---:|---:|---:|---|---:|---:|
| 1 | 905.31 | 821.88 | 1.102 | yes | 0.283 | 0.55 |
| 16 | 1020.74 | 1021.08 | 1.000 | yes | 1.379 | 1.56 |
| 64 | 1035.78 | 1034.40 | 1.001 | yes | 1.407 | 1.55 |
| 256 | 1471.91 | 1338.45 | 1.100 | yes | 2.354 | 1.20 |

- **Flat at M = 16 and 64; about 10% slower at M = 1 and 256.** This recorder
  stores only medians, so it gives no window ranges for these rows. The M = 256
  cost does repeat in the K and N packets, where every bounded window at
  256 × 12288 is slower than every static window. Their member attribution puts
  it in the gate product (see `../gfx1151_n_bounded_quantized_mlp_20261101/`).
  The M = 1 result is not attributed.
- **Setup.** One bounded compile took 2.12 s, and its owner holds 18 device
  buffers and 24.7 MB at capacity. Each static M costs its own 0.6–1.6 s compile.
- **Bitwise equality with static-at-M** was observed at these shapes. It is not
  asserted in general.

## Limits

- **Diagnostic only; no promotion.** HIP events on WSL2 are not an admitted
  kernel-side witness. Profiler counters are **unverified**: there is no
  `/dev/kfd`, so `rocprofv3` returns no records.
- **gfx1151 only.** These are gfx1151 numbers, not parity with gfx1201's packet.
  The two parts differ in product instruction (F16-decode against native FP8
  WMMA), image and memory system, so no ratio between the packets is drawn.
- **No `tile_check`.** gfx1201's `tile_check` isolated a capacity-selected
  `[32, 32]` register panel that made the bounded gate product slow at small M.
  On gfx1151 every bounded owner whose tile was recorded took `[16, 32]` at
  capacity 256 (`gate_product_macro_tile` in the K, N and H `members.json`), so
  that small-M mechanism does not occur. The opposite one does: see the H packet.
