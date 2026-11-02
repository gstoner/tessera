# H-bounded quantized MLP vs static programs on gfx1151 (2026-11-01)

**Sync key:** `GFX1151-BOUNDED-QUANTIZED-H-20261101` (program `GFX1151-QUANTIZED-MLP-TIMING-20261101`).
**Owner:** ROCM-NVFP4-INGEST-1. The gfx1151 counterpart of
`../gfx1201_h_bounded_quantized_mlp_20261029/`.

**Host:** Princess-Luna, Radeon 8060S (gfx1151, RDNA 3.5), Ubuntu 26.04 under
WSL2, ROCm 10.0 / HIP 7.15.26333, apt LLVM/MLIR 23.1.1 (assertions OFF) `build/`
tree. Ran unsandboxed under `flock /tmp/tessera-gfx1151-device.lock`, and each
recorder checked the lock was held.

**Files:**
- `paired.json`, from `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --h-bound --bounds 256 512 --kn 4096 256`.
- `members.json`, from `benchmarks/rocm/attribute_bounded_quantized_mlp_members.py`.
  It generalizes the gfx1201 packet-local `member_attribution.py`.

Both ran at commit `070f445d7` on a clean worktree, with a never-timed warm-up
owner first, and record their hashes, `route`, `latency_source` and
`timing_source`.

## Method

One v8 program was compiled with `m_bound=256, h_bound=512` at K/N = 4096/256 and
held in one prepared owner, which takes a full update to each active (M, H). The
control is the static v4 program at that shape. Timing is the HIP-event device
time per invocation (`hip_event_owner_window`): windows of 50, 9 rounds, with arm
and shape order alternating. Outputs were compared bitwise first.

## Result (median ms per invocation; window range in brackets)

| Active (M, H) | H-bounded owner | Static | Bounded / static | Bitwise equal | Static compile s |
|---|---|---|---:|---|---:|
| 16 × 128 | 0.3633 [0.3574–0.3639] | 0.3667 [0.3618–0.3673] | 0.991 | yes | 0.57 |
| 16 × 512 | 0.4485 [0.4445–0.4521] | 0.4461 [0.4428–0.4505] | 1.006 | yes | 1.56 |
| 64 × 256 | 0.4241 [0.4219–0.4263] | 0.4176 [0.4168–0.4209] | 1.016 | yes | 1.74 |
| 256 × 128 | 0.4949 [0.4910–0.5014] | 0.4705 [0.4668–0.4773] | 1.052 | yes | 1.71 |
| 256 × 512 | 1.1770 [1.1676–1.2398] | 1.0808 [1.0688–1.1048] | 1.089 | yes | 1.71 |

- **M = 16 is flat**, unlike gfx1201, where the capacity-selected `[32, 32]`
  panel made the bounded gate product 4–5× slower at M = 16. On gfx1151 the
  bounded owner took `[16, 32]` at capacity (`gate_product_macro_tile`), which is
  the same panel the static programs take at M = 16.
- **A real cost at larger M.** At 64 × 256, 256 × 128 and 256 × 512 every bounded
  window is slower than every static window, by 2%, 5% and 9%.
- **Setup.** One bounded compile took 1.98 s, and its owner holds 18 device
  buffers and 17.1 MB at capacity.

## Per-member device windows (`members.json`)

Each member is timed alone through `invoke_range`: 200-invocation windows,
7 rounds, arm order alternating. Values are median µs, in the order gate
quantizer / gate-up expansion / gate product / SiLU / down quantizer / down
expansion / down product.

| (M, H, K, N) | Arm | Gate tile | Members (µs) |
|---|---|---|---|
| 16 × 128 × 4096 × 256 | bounded | [16, 32] | 4.15, 21.24, **314.21**, 2.30, 3.81, 3.61, 15.93 |
| 16 × 128 × 4096 × 256 | static | [16, 32] | 3.97, 20.09, **314.10**, 1.57, 3.77, 2.33, 16.21 |
| 16 × 512 × 4096 × 256 | bounded | [16, 32] | 4.16, 73.52, **316.32**, 2.43, 3.82, 4.43, 44.76 |
| 16 × 512 × 4096 × 256 | static | [16, 32] | 3.99, 72.66, **314.95**, 1.60, 3.77, 3.63, 44.79 |
| 256 × 512 × 4096 × 256 | bounded | [16, 32] | 25.24, 73.83, **1030.36**, 6.30, 5.22, 4.70, 57.90 |
| 256 × 512 × 4096 × 256 | static | **[32, 32]** | 25.55, 73.52, **915.37**, 3.84, 5.15, 3.78, 53.68 |

- **The opposite of the gfx1201 tile effect.** At 256 × 512 the static program
  picks the 32-row `[32, 32]` panel and its gate product is 12.6% faster than the
  bounded owner's `[16, 32]` (915 against 1030 µs). On gfx1201 the larger panel,
  chosen at capacity, was the slow one at small M. Here it is the fast one at
  large M, and the bounded owner does not have it.
- **The runtime-H elementwise members cost about 1 µs** (SiLU 2.3 against 1.6;
  down expansion 3.6 against 2.3 at H = 128). That is small next to the product.
- **Not attributed:** the 5% at 256 × 128, where no member windows were taken.
  The N packet shows a same-panel gate-product cost at M = 256, so the two are
  probably related, but this is not established.

## Limits

- **Diagnostic only; no promotion.** HIP events on WSL2 are not an admitted
  kernel-side witness. Profiler counters are **unverified** (no `/dev/kfd`).
- **No `tile_check`.** On gfx1151 the bounded owner already takes the small panel
  at capacity, so gfx1201's capacity-16 versus capacity-256 isolation has nothing
  to isolate. Whether a capacity-chosen `[32, 32]` bounded image would recover
  the large-M cost was not measured.
- **gfx1151 only.** No parity with gfx1201 and no ratio between the packets.
- **Bitwise equality with static** was observed at these shapes. It is not
  asserted in general.
