# K-bounded quantized MLP vs static programs on gfx1201 (2026-10-28)

**Sync key:** `GFX1201-BOUNDED-QUANTIZED-K-20261028`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

`paired.json` was produced by `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py` at commit `171951af1` on a clean worktree. It records the recorder, compiler and HIP provider hashes.

## What was measured

One program compiled with `compile_native_quantized_program(..., m_bound=256, k_bound=12288)` at H/N = 128/256, held in one prepared owner. For each active (M, K), that owner takes a full update to new weights at that K. The control is the static v4 program compiled at exactly that (M, K), in its own owner.

**Method:**
- HIP-event device time per program invocation; every invocation runs all seven members, including both weight expansions, in both arms.
- Windows of 50 invocations, 9 rounds; the arm order and the (M, K) order alternate every round. The bounded update precedes its window and is not timed.
- Outputs are compared bitwise before timing; the recorder aborts on any difference.

## Result (median ms per invocation)

| Active (M, K) | K-bounded owner | Static at (M, K) | Bounded / static | Bitwise equal | Static compile s |
|---|---|---|---|---|---|
| 16 × 4096 | 0.0656 | 0.0651 | 1.007 | yes | 1.00 |
| 16 × 12288 | 0.1489 | 0.1445 | 1.030 | yes | 0.64 |
| 64 × 8192 | 0.1116 | 0.1082 | 1.032 | yes | 1.00 |
| 256 × 4096 | 0.0823 | 0.0820 | 1.004 | yes | 0.95 |
| 256 × 12288 | 0.1899 | 0.1870 | 1.015 | yes | 0.59 |

- **A small, real cost.** At 16 × 12288, 64 × 8192 and 256 × 12288 every bounded window is slower than every static window. At 16 × 4096 one bounded window was an outlier (0.16 ms); its median is unaffected.
- **Setup.** One bounded compile took 1.17 s, and its owner holds 18 device buffers and 24.7 MB at capacity. Each static (M, K) costs its own compile of 0.6–1.0 s.

## Where the cost is (per-member device windows, 200 repeats, 7 interleaved rounds)

`profile_members()` at two shapes, median µs per member, gate quantizer / gate expansion / gate product / SiLU-mul / down quantizer / down expansion / down product:

| (M, K) | Arm | Members (µs) |
|---|---|---|
| 16 × 12288 | bounded | 4.27, 16.88, **113.43**, 1.65, 3.96, 2.86, 4.27 |
| 16 × 12288 | static | 4.35, 16.71, **108.94**, 1.59, 3.94, 2.79, 4.44 |
| 64 × 8192 | bounded | 6.05, 12.64, **82.59**, 1.69, 4.01, 2.74, 4.23 |
| 64 × 8192 | static | 5.77, 12.28, **79.39**, 1.67, 4.00, 2.83, 4.48 |

- **The gate product carries it: about +4%.** Both arms use the same physical route (`gfx1201_register_wmma_blockscale_nk_1x2_k32`, global staging). The bounded image is the shape-free projection that reads M, N and K at launch; the static image has constant loop bounds. This is the same mechanism as the arbitrary-M product cost in the v5 packet (`../gfx1201_bounded_quantized_mlp_20261027/`, 0.997–1.032×).
- **The runtime-K expansion is within 1–3%** of the static expansion, and the other members are flat.
- This attribution is from one diagnostic run of capture windows; it is not a separate timing claim.

## Limits

- **Diagnostic only.** HIP events on WSL2 are not an admitted kernel-side witness, `rocprofv3` returns no records without `/dev/kfd`, and nothing here is a promotion.
- **One H/N, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with static-at-(M, K) was observed at these shapes and in the device tests.** It is not asserted in general.
