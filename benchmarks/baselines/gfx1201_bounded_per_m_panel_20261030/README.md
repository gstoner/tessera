# Per-extent product images for bounded quantized MLPs on gfx1201 (2026-10-30)

**Sync key:** `GFX1201-BOUNDED-PER-M-PANEL-20261030`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

**Recorded by:** `per_m_timing.py` (`per_m.json`) and `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --h-bound --bounds 256 2048 --kn 4096 256` (`paired_h.json`), both at commit `39a4b1ee3` on a clean worktree.

## What changed

A bounded product used to run one image across every admitted shape. Whichever panel that image used lost somewhere:

| Bounded program | Single image | Where it lost |
|---|---|---|
| M-only, H 2048 | LDS `[128, 128]` | gate product 2.9× static at M = 16 |
| H-bound, capacity (256, 2048) | global `[16, 32]` | whole program 2.75× at 256 × 2048 (`../gfx1201_bounded_global_panel_20261030/`) |

Now a bounded product carries:

- **A base image.** The register panel for M >= 1; a runtime-K/N/H product's base is the global shape-free image.
- **One LDS image per box where the static rule takes an LDS panel.** A box is an M range and, when the product varies K or N, a range of that extent. Each variant is compiled with the panel evaluated at its box's corner (`tessera.rocm.blockscale_min_m`, `_at_n`, `_at_k`, read by the native selector, which stays the authority).
  - A variant whose N varies at launch masks its column edge (`arbitrary-n`, `whole_n = false`).
- **One HIP owner that selects the image for the active extents.**

Register panels are not split further, because 16 vs 32 rows measured mixed.

## M-only program, H/K/N 2048/4096/256, capacity M = 256 (`per_m.json`)

Gate variants: 97–128 → LDS `[128, 64]`, 129–256 → LDS `[128, 128]`.

| Active M | Gate image | Static gate tile | Whole program bounded / static | Gate product bounded / static (µs) | Bitwise equal |
|---|---|---|---|---|---|
| 16 | base | `[16, 32]` | 1.011 | 43.4 / 42.6 | yes |
| 64 | base | `[32, 32]` | 0.969 | 53.7 / 62.0 | yes |
| 100 | 1 | `[128, 64]` | 1.002 | 129.0 / 128.5 | yes |
| 128 | 1 | `[128, 64]` | 1.010 | 128.8 / 126.9 | yes |
| 200 | 2 | `[128, 128]` | 1.032 | 193.2 / 181.9 | yes |
| 256 | 2 | `[128, 128]` | 1.102 | 241.7 / 205.1 | yes |

## H-bound program, capacity (256, 2048) at K/N 4096/256 (`paired_h.json`)

The gate product's N is 2H, so its boxes run over (M, H/16). There are four boxes: 97–127 × 32–128 and 128 × 127–128 take `[128, 64]`; 129–256 × 63–124 takes `[128, 64]`, and 129–256 × 125–128 takes `[128, 128]`.

| Active (M, H) | Bounded owner (ms) | Static (ms) | Bounded / static | Before (global base only) | Bitwise equal |
|---|---|---|---|---|---|
| 16 × 512 | 0.0874 | 0.0841 | 1.040 | 1.032 | yes |
| 16 × 2048 | 0.2017 | 0.1912 | 1.055 | 1.024 | yes |
| 100 × 1024 | 0.1983 | 0.1960 | 1.012 | 1.116 | yes |
| 256 × 512 | 0.1285 | 0.1287 | 0.998 | 0.999 | yes |
| 256 × 2048 | 0.4002 | 0.3474 | **1.152** | **2.750** | yes |
| 200 × 1008 (N ends mid-tile) | 0.2172 | 0.2166 | 1.003 | — | yes |

- **Both programs now track static across the whole range: 0.97–1.15×.** The capacity corner fell from 2.75× to 1.15×. One bounded compile now takes 1.96 s (it builds the variant images); the owner holds 50.8 MB at capacity.
- **An intermediate M-only selection (before boxes) mis-ran the H program at 256 × 512 (1.56×).** At M = 256 it chose LDS from the capacity N, while static at N = 1024 takes a register panel. Boxes over (M, H) fixed it (0.998×). That intermediate run is not kept.
- **The remaining 10–15% at the capacity corner is the shape-free LDS image itself.** It masks rows and, for the H program, columns, against static's whole-tile image at exactly that shape: gate product 241.7 vs 205.1 µs at M = 256 in the M-only program. An earlier diagnostic run (not recorded here) measured the M-only program's single LDS image at 1.13× at M = 256, so this is the arbitrary-M masking, not variant selection.

## Limits

- **Diagnostic only.** HIP events on WSL2 are not an admitted kernel-side witness, and nothing here is a promotion.
- **The variant boxes are the static rule's.** Where that rule is itself not the fastest choice (16 vs 32 rows, or LDS at the edge of its envelope), the bounded program inherits it.
- **One K and N per program, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with the static program was observed at these shapes and in the device tests.** It is not asserted in general.
