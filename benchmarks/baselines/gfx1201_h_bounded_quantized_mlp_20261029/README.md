# H-bounded quantized MLP vs static programs on gfx1201 (2026-10-29)

**Sync key:** `GFX1201-BOUNDED-QUANTIZED-H-20261029`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

**Recorded by:** `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --h-bound --bounds 256 512 --kn 4096 256` (`paired.json`, `paired_ec797f262.json`), `member_attribution.py` (`members.json`, `members_ec797f262.json`) and `tile_check.py` (`tile_check.json`).

There are two runs:

- **`ec797f262`, before.** The first H-bounded program. The capacity schedule chose its gate-product panel at capacity, and this run found the resulting small-M cost.
- **`932ac0420`, after.** A shape-free product now chooses its W8A8 register panel for the rows it admits, M >= 1 (`tessera.rocm.blockscale_min_m`, §"Bounded-hidden v8 MLP").

Both ran on a clean worktree. Each JSON records the recorder, compiler and HIP provider hashes.

## What was measured

One v8 program compiled with `compile_native_quantized_program(..., m_bound=256, h_bound=512)` at K = 4096 and N = 256, held in one prepared owner. For each active (M, H), that owner takes a full update to new weights at that H. The control is the static v4 program compiled at exactly that (M, H), in its own owner.

**Method:** the same as the K and N packets (`../gfx1201_k_bounded_quantized_mlp_20261028/`):
- HIP-event device time per invocation of all seven members.
- Windows of 50 invocations, 9 rounds, with arm and shape order alternating each round. The update is not timed.
- Outputs are checked bitwise before timing.

## Result (median ms per invocation)

| Active (M, H) | Before: bounded | Before: / static | After: bounded | After: static | After: / static | Bitwise equal |
|---|---|---|---|---|---|---|
| 16 × 128 | 0.2432 | 3.907 | 0.0652 | 0.0624 | 1.045 | yes |
| 16 × 512 | 0.2638 | 3.091 | 0.0891 | 0.0862 | 1.035 | yes |
| 64 × 256 | 0.0860 | 1.235 | 0.0741 | 0.0712 | 1.040 | yes |
| 256 × 128 | 0.0908 | 1.197 | 0.0784 | 0.0766 | 1.023 | yes |
| 256 × 512 | 0.1284 | 1.014 | 0.1234 | 0.1257 | 0.982 | yes |

- **After: 0.98–1.05× static at every shape**, bitwise equal at all five. At 16 × 128 one bounded window was an outlier (0.173 ms); its median is unaffected.
- **Setup.** One bounded compile took 1.15 s, and its owner holds 18 device buffers and 17.1 MB at capacity. Each static shape costs its own compile of 0.9–1.0 s.

## Where the cost was, and what moved

Per-member device windows (median µs per member invocation, 200-invocation windows, 7 alternating rounds). Members: gate quantizer / gate/up expansion / gate product / SiLU / down quantizer / down expansion / down product.

| (M, H) | Arm | Members (µs) |
|---|---|---|
| 16 × 128 | before, bounded | 4.17, 7.33, **217.03**, 2.61, 4.10, 3.77, 4.54 |
| 16 × 128 | after, bounded | 4.19, 7.34, **39.40**, 2.56, 4.16, 3.68, 4.41 |
| 16 × 128 | static | 4.18, 7.36, **38.55**, 1.88, 4.16, 2.92, 4.45 |
| 256 × 512 | before, bounded | 11.54, 23.40, **65.33**, 4.01, 4.16, 3.64, 9.19 |
| 256 × 512 | after, bounded | 11.69, 24.24, **62.62**, 4.19, 4.13, 3.56, 9.19 |
| 256 × 512 | static | 12.91, 23.67, **65.45**, 2.74, 4.07, 3.11, 8.94 |

(`members.json` also has 16 × 512. The static rows are from the after run; the before run's static rows are in `members_ec797f262.json`.)

- **The cost was the gate product's panel, chosen at capacity.** At capacity (M = 256, N = 2·512) the rule takes the 32-row `[32, 32]` register panel. Static programs at M = 16 take `[16, 32]`. `tile_check.json` (run at `ec797f262`) isolated it: the same v8 program compiled with row capacity 16 got `[16, 32]` and ran its gate product at 1.00× / 1.03× static at M = 16 (H = 128 / 512). Row capacity 256 got `[32, 32]` and ran at 4.56× / 4.27×. All four cases were bitwise equal. At M = 16 both tiles launch the same grid of one-wave workgroups (one row block; 8 workgroups at H = 128, 32 at H = 512), so this is per-wave loop cost at low occupancy, not grid size.
- **The fix chooses the panel at the smallest admitted M.** With M >= 1 that is the 16-row panel. The bounded gate product now matches static at M = 16 (39.4 vs 38.6 µs). At 256 × 512 it is slightly faster than static's own 32-row panel (62.6 vs 65.5 µs).
- **What the fix can cost elsewhere.** A separate paired check (static programs, 16-row forced against the rule's 32-row) across the band where the 32-row rule fires for K16 scales measured 0.78–1.33×. So a shape-free product whose capacity sits in that band can be up to about 1.33× slower at the capacity end, in exchange for not being 4–5× slower at small M. The static selector is unchanged: that check is mixed, so it is not grounds to change the static rule.
- **The H members themselves cost at most about 1 µs.** The runtime-width SiLU (+0.7 µs at M = 16) and the runtime-K down expansion (+0.4–0.8 µs) are slightly slower than their static counterparts. The runtime-N gate/up expansion, the down quantizer and the down product are within noise.

## Limits

- **Diagnostic only.** HIP events on WSL2 are not an admitted kernel-side witness, `rocprofv3` returns no records without `/dev/kfd`, and nothing here is a promotion.
- **Absolute static times move between runs.** The static gate product at 16 × 128 read 38.5 µs in one run and 60.4 µs in another. Only the ratios measured within one paired run are compared here.
- **The panel choice covers only the register panel.** A capacity that selects an LDS panel keeps the capacity rule (and the runtime K, N and H images refuse LDS products). One panel still serves the whole range; selecting among measured images per active shape remains open.
- **One K and N, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with the static program was observed at these shapes and in the device tests.** It is not asserted in general.
