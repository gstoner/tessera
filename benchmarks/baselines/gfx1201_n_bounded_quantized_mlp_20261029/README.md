# K- and N-bounded quantized MLP vs static programs on gfx1201 (2026-10-29)

**Sync key:** `GFX1201-BOUNDED-QUANTIZED-N-20261029`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

`paired.json` was produced by `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --n-bound 512` at commit `98822e8a4` on a clean worktree. It records the recorder, compiler and HIP provider hashes.

## What was measured

One v7 program compiled with `compile_native_quantized_program(..., m_bound=256, k_bound=12288, n_bound=512)` at H = 128, held in one prepared owner. For each active (M, K, N), that owner takes a full update to new weights at that K and N. The control is the static v4 program compiled at exactly that (M, K, N), in its own owner.

**Method:** the same as the K packet (`../gfx1201_k_bounded_quantized_mlp_20261028/`):
- HIP-event device time per invocation of all seven members.
- Windows of 50 invocations, 9 rounds, with arm and shape order alternating each round. The update is not timed.
- Outputs are checked bitwise before timing.

## Result (median ms per invocation)

| Active (M, K, N) | Bounded owner | Static at shape | Bounded / static | Bitwise equal | Static compile s |
|---|---|---|---|---|---|
| 16 × 4096 × 256 | 0.0666 | 0.0653 | 1.020 | yes | 1.01 |
| 16 × 4096 × 512 | 0.0668 | 0.0650 | 1.029 | yes | 0.55 |
| 64 × 8192 × 384 | 0.1120 | 0.1084 | 1.033 | yes | 1.00 |
| 256 × 12288 × 128 | 0.1908 | 0.1869 | 1.021 | yes | 1.02 |
| 256 × 12288 × 512 | 0.1906 | 0.1866 | 1.021 | yes | 0.49 |

- **A small, real cost: 2–3%.** At every shape but the first, every bounded window is slower than every static window. At 16 × 4096 × 256 one bounded window was an outlier (0.165 ms); its median is unaffected.
- **Setup.** One bounded compile took 1.14 s, and its owner holds 18 device buffers and 25.0 MB at capacity. Each static shape costs its own compile of 0.5–1.0 s.

## Where the cost is (per-member device windows, 200 repeats, 7 interleaved rounds)

Median µs per member: gate quantizer / gate expansion / gate product / SiLU-mul / down quantizer / down expansion / down product.

| (M, K, N) | Arm | Members (µs) |
|---|---|---|
| 64 × 8192 × 384 | bounded | 6.26, 11.94, **80.02**, 1.64, 3.92, 3.00, 4.29 |
| 64 × 8192 × 384 | static | 6.19, 12.11, **77.52**, 1.67, 3.97, 2.81, 4.44 |
| 256 × 12288 × 128 | bounded | 27.73, 15.67, **121.94**, 1.82, 3.94, 2.79, 4.21 |
| 256 × 12288 × 128 | static | 29.28, 15.54, **118.35**, 1.73, 3.90, 2.95, 4.52 |

- **The gate product carries it again: about +3%**, the shape-free gate image against a static image with constant loop bounds. This is the cost already recorded in the K packet.
- **The N members cost nothing measurable.** The runtime-N down expansion and the shape-free down product are within noise, and the bounded down product is slightly faster at both shapes.
- This attribution is from one diagnostic run of capture windows; it is not a separate timing claim.

## Limits

- **Diagnostic only.** HIP events on WSL2 are not an admitted kernel-side witness, `rocprofv3` returns no records without `/dev/kfd`, and nothing here is a promotion.
- **One H, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with the static program was observed at these shapes and in the device tests.** It is not asserted in general.
