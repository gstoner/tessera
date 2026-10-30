# H-bounded quantized MLP vs static programs on gfx1201 (2026-10-29)

**Sync key:** `GFX1201-BOUNDED-QUANTIZED-H-20261029`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

`paired.json` was produced by `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --h-bound --bounds 256 512 --kn 4096 256` at commit `ec797f262` on a clean worktree. It records the recorder, compiler and HIP provider hashes. `members.json` and `tile_check.json` come from the two diagnostic scripts in this directory, run at the same commit.

## What was measured

One v8 program compiled with `compile_native_quantized_program(..., m_bound=256, h_bound=512)` at K = 4096 and N = 256, held in one prepared owner. For each active (M, H), that owner takes a full update to new weights at that H. The control is the static v4 program compiled at exactly that (M, H), in its own owner.

**Method:** the same as the K and N packets (`../gfx1201_k_bounded_quantized_mlp_20261028/`):
- HIP-event device time per invocation of all seven members.
- Windows of 50 invocations, 9 rounds, with arm and shape order alternating each round. The update is not timed.
- Outputs are checked bitwise before timing.

## Result (median ms per invocation)

| Active (M, H) | Bounded owner | Static at shape | Bounded / static | Bitwise equal | Static compile s |
|---|---|---|---|---|---|
| 16 × 128 | 0.2432 | 0.0623 | 3.907 | yes | 1.02 |
| 16 × 512 | 0.2638 | 0.0853 | 3.091 | yes | 0.94 |
| 64 × 256 | 0.0860 | 0.0697 | 1.235 | yes | 1.00 |
| 256 × 128 | 0.0908 | 0.0759 | 1.197 | yes | 0.91 |
| 256 × 512 | 0.1284 | 0.1266 | 1.014 | yes | 0.95 |

- **Correct everywhere, but slow at small M.** Every output is bitwise equal to the static program. At 256 × 512 the bounded owner costs 1.4%; at M = 16 it costs 3–4×. The bounded and static ranges do not overlap at any shape.
- **Setup.** One bounded compile took 1.50 s, and its owner holds 18 device buffers and 17.1 MB at capacity. Each static shape costs its own compile of 0.9–1.0 s.

## Where the cost is

Per-member device windows (`members.json`, median µs per member invocation, 200-invocation windows, 7 alternating rounds). Members: gate quantizer / gate/up expansion / gate product / SiLU / down quantizer / down expansion / down product.

| (M, H) | Arm | Members (µs) |
|---|---|---|
| 16 × 128 | bounded | 4.17, 7.33, **217.03**, 2.61, 4.10, 3.77, 4.54 |
| 16 × 128 | static | 4.26, 7.25, **38.50**, 1.84, 4.17, 2.82, 4.43 |
| 16 × 512 | bounded | 4.30, 23.80, **214.16**, 2.57, 3.97, 3.69, 8.80 |
| 16 × 512 | static | 4.23, 24.00, **38.44**, 1.93, 4.24, 3.20, 8.36 |
| 256 × 512 | bounded | 11.54, 23.40, **65.33**, 4.01, 4.16, 3.64, 9.19 |
| 256 × 512 | static | 12.63, 23.21, **64.87**, 2.84, 4.13, 3.20, 8.97 |

- **The gate product carries almost all of it.** The bounded gate product is 5.6× the static one at M = 16 and matches it at 256 × 512.
- **The cause is the macro tile the capacity program selects, not the H machinery.** The capacity schedule (M = 256, N = 2·512) picks a `[32, 32]` gate-product tile. The static schedule picks `[16, 32]` at M = 16 and at 256 × 128, and `[32, 32]` at 256 × 512, the one shape where the two arms match. `tile_check.json` isolates it: at M = 16 the same v8 program compiled with row capacity 16 gets a `[16, 32]` tile and runs its gate product at 1.00× / 1.03× static (H = 128 / 512). Row capacity 256 gets `[32, 32]` and runs at 4.56× / 4.27×. All four cases are bitwise equal. At M = 16 both tiles launch the same grid of one-wave workgroups (one row block; 8 workgroups at H = 128, 32 at H = 512), so this is per-wave loop cost at low occupancy, not grid size.
- **The H members themselves cost at most about 1 µs.** The runtime-width SiLU (+0.7 µs at M = 16) and the runtime-K down expansion (+0.4–1.0 µs) are slightly slower than their static counterparts. The runtime-N gate/up expansion, the down quantizer and the down product are within noise.
- **What it implies.** A capacity program uses one product tile for the whole admitted range. For a wide M range that tile should be chosen for the range, or selected per active shape among measured images. That is open work, recorded with the LDS-staged capacity product follow-up.

## Limits

- **Diagnostic only.** HIP events on WSL2 are not an admitted kernel-side witness, `rocprofv3` returns no records without `/dev/kfd`, and nothing here is a promotion.
- **Absolute static times move between runs.** The static gate product at 16 × 128 read 38.5 µs in one run and 60.4 µs in the other. Only the ratios measured within one paired run are compared here.
- **One K and N, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with the static program was observed at these shapes and in the device tests.** It is not asserted in general.
