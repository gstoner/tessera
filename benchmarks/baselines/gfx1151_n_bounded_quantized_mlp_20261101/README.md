# N-bounded quantized MLP vs static programs on gfx1151 (2026-11-01)

**Sync key:** `GFX1151-BOUNDED-QUANTIZED-N-20261101` (program `GFX1151-QUANTIZED-MLP-TIMING-20261101`).
**Owner:** ROCM-NVFP4-INGEST-1. The gfx1151 counterpart of
`../gfx1201_n_bounded_quantized_mlp_20261029/`.

**Host:** Princess-Luna, Radeon 8060S (gfx1151, RDNA 3.5), Ubuntu 26.04 under
WSL2, ROCm 10.0 / HIP 7.15.26333, apt LLVM/MLIR 23.1.1 (assertions OFF) `build/`
tree. Ran unsandboxed under `flock /tmp/tessera-gfx1151-device.lock`, and each
recorder checked the lock was held.

**Files:**
- `paired.json`, from `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --n-bound 512`.
- `members.json`, from `benchmarks/rocm/attribute_bounded_quantized_mlp_members.py`.

Both ran at commit `070f445d7` on a clean worktree, with a never-timed warm-up
owner first, and record their hashes, `route`, `latency_source` and
`timing_source`.

## Method

One program was compiled with `m_bound=256, k_bound=12288, n_bound=512` at
H = 128 and held in one prepared owner, which takes a full update to each active
(M, K, N). The control is the static v4 program at that shape. Timing is the
HIP-event device time per invocation (`hip_event_owner_window`): windows of 50,
9 rounds, with arm and shape order alternating. Outputs were compared bitwise
first.

## Result (median ms per invocation; window range in brackets)

| Active (M, K, N) | N-bounded owner | Static | Bounded / static | Bitwise equal | Static compile s |
|---|---|---|---:|---|---:|
| 16 × 4096 × 256 | 0.3563 [0.3552–0.3621] | 0.3573 [0.3563–0.3624] | 0.997 | yes | 0.55 |
| 16 × 4096 × 512 | 0.3557 [0.3552–0.3617] | 0.3572 [0.3560–0.3612] | 0.996 | yes | 0.95 |
| 64 × 8192 × 384 | 0.6875 [0.6861–0.6977] | 0.6929 [0.6926–0.7033] | 0.992 | yes | 1.92 |
| 256 × 12288 × 128 | 1.5078 [1.4523–1.5616] | 1.3672 [1.3229–1.5087] | 1.103 | yes | 1.81 |
| 256 × 12288 × 512 | 1.4750 [1.4575–1.5269] | 1.3490 [1.3337–1.4684] | 1.093 | yes | 0.88 |

- **Flat at M ≤ 64.** The windows overlap.
- **About 10% slower at M = 256.** The medians differ by 9–10%, but the windows
  overlap at the top of the static range, so this packet alone does not separate
  the arms. The K packet does at 256 × 12288 × 256, and the member attribution
  below does.
- **Setup.** One bounded compile took 2.16 s, and its owner holds 18 device
  buffers and 25.0 MB at capacity.

## Per-member device windows (`members.json`)

Each member is timed alone through `invoke_range`: 200-invocation windows,
7 rounds, arm order alternating. Values are median µs, in the order gate
quantizer / gate-up expansion / gate product / SiLU / down quantizer / down
expansion / down product.

| (M, H, K, N) | Arm | Members (µs) |
|---|---|---|
| 64 × 128 × 8192 × 384 | bounded | 12.65, 39.82, **621.25**, 1.59, 3.88, 2.64, 16.78 |
| 64 × 128 × 8192 × 384 | static | 12.74, 39.61, **619.86**, 1.61, 3.77, 2.53, 16.48 |
| 256 × 128 × 12288 × 128 | bounded | 68.49, 65.79, **1149.03**, 1.94, 3.83, 2.74, 16.95 |
| 256 × 128 × 12288 × 128 | static | 68.13, 62.84, **1047.64**, 2.05, 4.17, 2.59, 16.90 |

- **At M = 256 the cost is the gate product: +9.7%** (1149 against 1048 µs). The
  other members agree within a few µs. Both arms use the same `[16, 32]` register
  panel, so the tile is not the difference.
- **Not established.** The likely explanation, by analogy with gfx1201, is that
  the bounded image is the shape-free projection that reads M, N and K at launch,
  while the static image has constant loop bounds. That mechanism is not
  measured on gfx1151. No ISA or counter evidence was taken, and counters are
  unavailable.
- At 64 × 8192 × 384 the arms are equal.

## Limits

- **Diagnostic only; no promotion.** HIP events on WSL2 are not an admitted
  kernel-side witness. Profiler counters are **unverified** (no `/dev/kfd`).
- **gfx1151 only.** No parity with gfx1201 and no ratio between the packets.
- **Bitwise equality with static** was observed at these shapes. It is not
  asserted in general.
