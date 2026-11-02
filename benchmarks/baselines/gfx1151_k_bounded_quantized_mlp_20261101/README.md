# K-bounded quantized MLP vs static programs on gfx1151 (2026-11-01)

**Sync key:** `GFX1151-BOUNDED-QUANTIZED-K-20261101` (program `GFX1151-QUANTIZED-MLP-TIMING-20261101`).
**Owner:** ROCM-NVFP4-INGEST-1. The gfx1151 counterpart of
`../gfx1201_k_bounded_quantized_mlp_20261028/`.

**Host:** Princess-Luna, Radeon 8060S (gfx1151, RDNA 3.5), Ubuntu 26.04 under
WSL2, ROCm 10.0 / HIP 7.15.26333, apt LLVM/MLIR 23.1.1 (assertions OFF) `build/`
tree. Ran unsandboxed under `flock /tmp/tessera-gfx1151-device.lock`, and each
recorder checked the lock was held.

**Files:**
- `paired.json`, from `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py`.
- `members.json`, from `benchmarks/rocm/attribute_bounded_quantized_mlp_members.py`.

Both ran at commit `070f445d7` on a clean worktree, with a never-timed warm-up
owner first. Each records its recorder, compiler and provider hashes, `route`,
`latency_source` and `timing_source`.

## Method

One program was compiled with `m_bound=256, k_bound=12288` at H/N = 128/256 and
held in one prepared owner. For each active (M, K) it takes a full update. The
control is the static v4 program compiled at that (M, K), in its own owner.

Timing is the HIP-event device time per program invocation
(`hip_event_owner_window`): windows of 50 invocations, 9 rounds, with the arm
order and the shape order alternating each round. Outputs were compared bitwise
before timing.

## Result (median ms per invocation; window range in brackets)

| Active (M, K) | K-bounded owner | Static at (M, K) | Bounded / static | Bitwise equal | Static compile s |
|---|---|---|---:|---|---:|
| 16 × 4096 | 0.3623 [0.3614–0.3626] | 0.3620 [0.3614–0.3622] | 1.001 | yes | 0.52 |
| 16 × 12288 | 1.0200 [1.0184–1.0234] | 1.0209 [1.0195–1.0234] | 0.999 | yes | 1.08 |
| 64 × 8192 | 0.6987 [0.6971–0.6999] | 0.6981 [0.6957–0.6987] | 1.001 | yes | 1.62 |
| 256 × 4096 | 0.4959 [0.4901–0.5043] | 0.4752 [0.4609–0.4778] | 1.044 | yes | 1.56 |
| 256 × 12288 | 1.4668 [1.4559–1.5346] | 1.3373 [1.3186–1.3821] | 1.097 | yes | 0.97 |

- **Flat at M = 16 and 64.** The windows overlap.
- **A real cost at M = 256.** At 256 × 4096 and 256 × 12288 every bounded window
  is slower than every static window, by 4% and 10%.
- **Setup.** One bounded compile took 2.35 s, and its owner holds 18 device
  buffers and 24.7 MB at capacity.

## Per-member device windows (`members.json`)

Each member is timed alone through `invoke_range` (`hip_event_member_range_window`):
200-invocation windows, 7 rounds, arm order alternating. Values are median µs.
Member order: gate quantizer / gate-up expansion / gate product / SiLU / down
quantizer / down expansion / down product.

| (M, H, K, N) | Arm | Members (µs) |
|---|---|---|
| 16 × 128 × 12288 × 256 | bounded | 6.01, 61.21, **929.85**, 1.70, 3.81, 2.51, 16.02 |
| 16 × 128 × 12288 × 256 | static | 5.96, 61.50, **930.21**, 1.63, 3.84, 2.41, 16.29 |
| 64 × 128 × 8192 × 256 | bounded | 12.71, 39.81, **622.12**, 1.66, 3.79, 2.42, 15.92 |
| 64 × 128 × 8192 × 256 | static | 12.80, 39.78, **621.66**, 1.63, 3.85, 2.38, 16.24 |

- **The gate product dominates**, at about 90% of the program on gfx1151, and it
  is equal between the arms at these two shapes. Both arms use the `[16, 32]`
  register panel. The runtime-K expansion matches the static one.
- These shapes reproduce gfx1201's attribution set. The M = 256 cost in the table
  above is attributed in the N packet, at 256 × 128 × 12288 × 128.

## Limits

- **Diagnostic only; no promotion.** HIP events on WSL2 are not an admitted
  kernel-side witness. Profiler counters are **unverified**: there is no
  `/dev/kfd`, so `rocprofv3` returns no records.
- **One H/N, gfx1151 only.** No parity with gfx1201 and no ratio between the two
  packets: the product instruction (F16-decode against FP8 WMMA), image and
  memory system all differ.
- **Bitwise equality with static-at-(M, K)** was observed at these shapes. It is
  not asserted in general.
