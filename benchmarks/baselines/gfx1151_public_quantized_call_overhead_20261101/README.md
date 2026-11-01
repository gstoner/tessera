# Public quantized MLP call overhead on gfx1151 (2026-11-01)

**Sync key:** `GFX1151-PUBLIC-QUANTIZED-CALL-OVERHEAD-20261101` (program `GFX1151-QUANTIZED-MLP-TIMING-20261101`).
**Owner:** ROCM-NVFP4-INGEST-1. The gfx1151 counterpart of the "after" columns of
`../gfx1201_public_quantized_call_overhead_20261028/`. That packet compared code
before and after the retained bounded owner. gfx1151 has only the current code,
so this packet has no before/after.

**Host:** Princess-Luna, Radeon 8060S (gfx1151, RDNA 3.5), Ubuntu 26.04 under
WSL2, ROCm 10.0 / HIP 7.15.26333, apt LLVM/MLIR 23.1.1 (assertions OFF) `build/`
tree. That tree has an empty `CMAKE_BUILD_TYPE`, so the host runtime libraries
are built without `-O`, which weighs on host-wall numbers. Ran unsandboxed under
`flock /tmp/tessera-gfx1151-device.lock`, and the recorder checked the lock was
held.

`paired.json` was produced by
`benchmarks/rocm/benchmark_public_bounded_quantized_mlp.py` at commit `070f445d7`
on a clean worktree, with a never-timed warm-up owner first. It records its
hashes, `route`, `latency_source` and `timing_source`
(`host_wall_perf_counter`).

## Method

H/K/N = 128/12288/256, capacity 256. Host wall per call, in windows of 10 calls,
12 rounds, with the arm order rotated and reversed each round; the table shows
the median. Outputs were compared bitwise first. The arms:

- **Public bounded:** a public `@jit` call through the bounded program. Since the
  2026-10-28 change it keeps one retained capacity owner across calls. The
  recorder's `timing` string still says it "prepares its owner every call",
  which predates that change.
- **Public static:** a public `@jit` call compiled at that M.
- **Held-owner control:** one prepared owner with an activation-only update,
  invoke and readback.

## Result (median ms per call)

| Active M | Public bounded | Public static | Held-owner control | Bounded / static | Control / bounded | Bitwise equal | Static cold first call s |
|---|---:|---:|---:|---:|---:|---|---:|
| 1 | 1.602 | 7.862 | 1.359 | 0.204 | 0.849 | yes | 1.34 |
| 16 | 2.575 | 8.922 | 2.322 | 0.289 | 0.902 | yes | 1.66 |
| 64 | 2.627 | 9.138 | 2.405 | 0.287 | 0.915 | yes | 1.66 |
| 256 | 4.485 | 11.964 | 4.154 | 0.375 | 0.926 | yes | 1.30 |

- **The public bounded call is 0.22–0.33 ms above the held-owner control.** The
  gfx1201 packet attributes that gap to the weight-byte comparison plus argument
  binding. It is not re-attributed on gfx1151.
- **The public static call costs 6.3–7.5 ms more than the bounded call.** It
  prepares an owner and uploads roots on every call.
- **Device time is part of every arm.** The program itself is about 0.8–1.5 ms on
  gfx1151 (see `../gfx1151_bounded_quantized_mlp_20261101/`), so the control is
  not mostly host cost at large M.

## Limits

- **Host wall on WSL2, diagnostic only; no promotion.** Profiler counters are
  **unverified** (no `/dev/kfd`).
- **gfx1151 only.** No parity with gfx1201 and no ratio between the packets. The
  host CPUs, runtime builds and device times all differ.
