# Per-extent product images for bounded quantized MLPs on gfx1201 (2026-10-30)

**Sync key:** `GFX1201-BOUNDED-PER-M-PANEL-20261030`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

**Recorded by:** `per_m_timing.py` (`per_m.json`), `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --h-bound --bounds 256 2048 --kn 4096 256` (`paired_h.json`) and `lds_attribution.py` (the attribution below).

- **The current results** were recorded at commit `ee2c2a302` on a clean worktree, behind a kept, never-timed warm-up owner.
- **The first run** (`per_m_39a4b1ee3.json`, `paired_h_39a4b1ee3.json`, commit `39a4b1ee3`) had no warm-up owner, and its LDS variants took K at launch everywhere. Its large-M bounded rows are biased; see "The first-owner artifact".

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
| 16 | base | `[16, 32]` | 1.007 | 43.4 / 42.9 | yes |
| 64 | base | `[32, 32]` | 0.962 | 52.4 / 61.9 | yes |
| 100 | 1 | `[128, 64]` | 0.992 | 127.3 / 128.4 | yes |
| 128 | 1 | `[128, 64]` | 1.005 | 126.9 / 126.7 | yes |
| 200 | 2 | `[128, 128]` | 1.008 | 183.0 / 182.4 | yes |
| 256 | 2 | `[128, 128]` | 1.012 | 204.9 / 202.2 | yes |

## H-bound program, capacity (256, 2048) at K/N 4096/256 (`paired_h.json`)

The gate product's N is 2H, so its boxes run over (M, H/16). There are four boxes: 97–127 × 32–128 and 128 × 127–128 take `[128, 64]`; 129–256 × 63–124 takes `[128, 64]`, and 129–256 × 125–128 takes `[128, 128]`.

| Active (M, H) | Bounded owner (ms) | Static (ms) | Bounded / static | Global base only | Bitwise equal |
|---|---|---|---|---|---|
| 16 × 512 | 0.0867 | 0.0841 | 1.031 | 1.032 | yes |
| 16 × 2048 | 0.1991 | 0.1937 | 1.028 | 1.024 | yes |
| 100 × 1024 | 0.2101 | 0.1950 | 1.077 | 1.116 | yes |
| 256 × 512 | 0.1237 | 0.1239 | 0.998 | 0.999 | yes |
| 256 × 2048 | 0.3599 | 0.3508 | **1.026** | **2.750** | yes |
| 200 × 1008 (N ends mid-tile) | 0.2170 | 0.2133 | 1.017 | — | yes |

- **Both programs now track static across the whole range: 0.96–1.08×**, bitwise equal everywhere. The H corner fell from 2.75× to 1.03×.
- **Run-to-run noise is about 5%.** 100 × 1024 read 1.012× in the first run and 1.077× here.
- **Setup.** One bounded compile now takes 2.1 s, because it builds the variant images. The owner holds 50.8 MB at capacity.
- **An intermediate M-only selection (before boxes) mis-ran the H program at 256 × 512 (1.56×).** At M = 256 it chose LDS from the capacity N, while static at N = 1024 takes a register panel. Boxes over (M, H) fixed it (0.998×). That intermediate run is not kept.

## The first-owner artifact, and what the shape-free image really costs

The first run put the bounded owner 10–18% behind static at M = 256 (gate product 241.7 vs 205.1 µs). `lds_attribution.py` takes the static MLP at 256 × 2048 and swaps only its gate-product image for images built from the same recipe. It then times that member alone, paired and interleaved:

| Gate image | No warm-up owner (µs) | Behind a warm-up owner (µs) |
|---|---|---|
| A: static checked image, prepared first | 242.1 | 208.5 |
| B: M and N at launch, whole tiles | 204.6 | 206.1 |
| C: B + K at launch | 211.2 | 215.0 |
| D: C + arbitrary M | 214.8 | 212.1 |
| E: D + arbitrary N | 211.5 | 212.2 |
| F: arbitrary M only (K fixed) | 206.3 | 206.8 |
| A2: static checked image, prepared last | 206.2 | 207.3 |

- **The first owner a process prepares ran about 17% slower for its whole life** (A vs A2, the same image). In the first run that owner was the bounded one. A kept, never-timed warm-up owner in that slot removes the effect, and both recorders now use one. The cause is not identified.
- **Reading M and N at launch, and masking M and N, cost nothing measurable** (B, F against A2).
- **Taking K at launch costs about 2–4%** (C, D, E). An LDS variant now takes K at launch only where its product's K varies: the K-bounded gate and the H-bounded down projection.

## Limits

- **Diagnostic only.** HIP events on WSL2 are not an admitted kernel-side witness, and nothing here is a promotion.
- **The first-owner effect is measured, not explained.** Any bounded-vs-static comparison without a warm-up owner may carry it.
- **The variant boxes are the static rule's.** Where that rule is itself not the fastest choice (16 vs 32 rows, or LDS at the edge of its envelope), the bounded program inherits it.
- **One K and N per program, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with the static program was observed at these shapes and in the device tests.** It is not asserted in general.
