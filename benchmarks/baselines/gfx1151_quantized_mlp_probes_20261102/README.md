# Native quantized MLP probes — gfx1151, 2026-11-02

**Sync key:** `GFX1151-QUANTIZED-MLP-PROBES-20261102`. **Owner:** ROCM-NVFP4-INGEST-1.
Probes the three open findings of `../gfx1151_quantized_mlp_20261101/` and its
bounded siblings (`GFX1151-QUANTIZED-MLP-TIMING-20261101`, PR #1031), and re-sweeps
the gfx1151 quantized MLP device tests on current main.

**Host:** Princess-Luna, a Radeon 8060S (gfx1151, Strix Halo, RDNA 3.5) on Ubuntu
26.04 under WSL2, with ROCm 10.0 / HIP 7.15 and the apt LLVM/MLIR 23.1.1
(assertions OFF) `build/` tree, rebuilt at commit `a4cc3b8b7` (main `d125bfd50`
plus `probe.py`) on a clean tree. Everything ran unsandboxed on that box. The device sweep and
every timed probe ran under `flock /tmp/tessera-gfx1151-device.lock`, which each
timed JSON records as held. The ISA mode only compiles and ran without it.

Every number is gfx1151-only and diagnostic. Nothing is compared with gfx1201 and
nothing is promoted. Profiler counters are unverified (WSL2 exposes no
`/dev/kfd`). The register and wait counts below are static counts from the
disassembly, not measurements.

## Method

`probe.py` (this directory) has three modes. Each timed mode prepares a
never-timed warm-up owner first and compares the bounded and static outputs
bitwise before timing. It then times 9 rounds of 200-invocation HIP-event windows,
alternating arm order each round, and keeps the median, min, max and every
window. Each JSON records `route`, `latency_source`, `timing_source`, and the
recorder, compiler and provider hashes. Every bounded/static pair below was
bitwise equal.

- `panel_min1.json` / `panel_min256.json`: the H-bounded owner (m_bound 256,
  h_bound 512, K/N 4096/256) against the static program at five active shapes.
  Each run times the gate product (member 2) alone and the whole program.
  `--force-min-m 256` makes this process choose every arbitrary-M register panel
  at capacity instead of at M = 1, through a **packet-local monkeypatch** of
  `admitted_rows_graph` / `verify_blockscale_schedule`. No production code
  changes. The run refuses unless the bounded gate product took the expected
  panel: `[32, 32]` forced and `[16, 32]` by default. The override applies to
  the down product too, so the `program` ratios include it.
- `rows.json`: the M-bounded owner (m_bound 256, H/K/N 128/12288/256) against
  static at M ∈ {1, 2, 8, 16, 32, 64, 128, 256}, with all seven members timed
  alone plus the program.
- `isa.json` + `isa/`: the gate-product HSACO of the bounded and static images at
  256 × 128 × 12288 × 256 (m_bound 256) and 256 × 128 × 12288 × 128 (M/K/N
  bounded). Both arms use `[16, 32]` there. The directory holds each
  image's `.hsaco`, its `llvm-objdump -d --mcpu=gfx1151` disassembly (`.s`), its
  `llvm-readelf --notes` (`.notes.txt`) and the backend IR.

## Device re-sweep (`device-tests.txt`)

56 passed, 6 failed. The run covered seven files: `test_native_quantized_mlp_program`,
`test_native_quantized_mlp_bounded_rows`, `test_public_quantized_mlp_{row,k,n,h}_bound_device`
and `test_quantized_mlp_per_m_panels_device`.

**All 6 failures are in `test_quantized_mlp_per_m_panels_device.py`,** and each one
is the same refusal: `native quantized MLP/product requires owning gfx1201`. That file
tests gfx1201's per-M LDS images (GFX1201-BOUNDED-PER-M-PANEL-20261030), which
gfx1151 does not have. It compiles for a pinned `target="rocm_gfx1201"` and has no
host gate. Because the refusal names a pinned arch rather than the host's,
`tests/conftest.py` keeps it as a failure by design. So this file cannot be
evaluated on gfx1151. The failures show a missing host gate in the test file,
not a gfx1151 regression and not a gfx1201 result. The other six files (56
tests) pass on gfx1151.

## Probe 1: a `[32, 32]` bounded panel chosen at capacity

Gate product (member 2), µs per invocation, median [min, max], bounded ÷ static:

| Active M × H | static tile | bounded `[16, 32]` (default) | bounded `[32, 32]` (forced) | static |
|---|---|---|---|---|
| 16 × 512 | `[16, 32]` | 317.7 [314.9, 318.6] — 1.004 | 350.5 [350.2, 350.7] — **1.107** | ≈316.5 |
| 64 × 512 | `[16, 32]` | 441.6 [410.3, 465.4] — 1.076 | 541.7 [526.9, 543.0] — **1.323** | ≈410 |
| 256 × 512 | `[32, 32]` | 1034.6 [965.4, 1043.6] — 1.126 | 924.2 [912.0, 961.9] — **1.003** | ≈920 |
| 256 × 256 | `[16, 32]` | 585.9 [564.8, 598.7] — 1.077 | 693.7 [684.0, 698.6] — **1.293** | ≈540 |
| 256 × 128 | `[16, 32]` | 425.0 [413.5, 436.3] — 1.078 | 536.0 [534.0, 537.7] — **1.361** | ≈394 |

Program ratios: default 1.010 / 1.071 / 1.115 / 1.058 / 1.063; forced
1.081 / 1.246 / 1.015 / 1.235 / 1.299. Each static column comes from its own run's
paired arm. The two runs' static medians agree within 1.5%.

**The `[32, 32]` panel recovers the one shape where static takes it (256 × 512:
1.126 → 1.003). It loses at every shape where static takes `[16, 32]`, and by
more than it gains: +10.7% at M = 16 and +29–36% at 256 × 256 and 256 × 128.**
So one panel chosen at capacity is not a fix on gfx1151. The register rule's
`[32, 32]` threshold depends on the active extent (full = (rows/32)·(n/32) ≥
256), so a single bounded image cannot match static across the range. The
measured direction is per-extent register-panel variants. Today gfx1151 builds
`ProductVariant`s only for LDS boxes, so it gets none. This is a measured
direction, not something built or proven here.

## Probe 2: same-panel cost at M = 256

Both gate products use `[16, 32]` and take identical 28-argument kernel
signatures. The disassembly has the same branch structure (34 branches at the
same relative layout) and almost the same instruction mix: 4 WMMA, 74 global
loads, 705 compares, 682 vs 684 `v_cndmask`, 46 exec-mask ops. What differs
(static counts, `isa/rows_256x128x12288x256_*`):

| | bounded | static |
|---|---|---|
| VGPRs | **181** | **255** |
| SGPRs | 78 | 88 |
| spills / scratch / LDS | 0 / 0 / 0 | 0 / 0 / 0 |
| `s_waitcnt` total | 47 | 35 |
| `s_waitcnt vmcnt(0)` (full drain) | **19** | **11** |

The bounded image keeps fewer registers live. It waits for all outstanding loads to
complete 8 more times than static does. That is consistent with fewer loads in flight
per wave, which is how a +8–9% gate-product cost could arise at an equal
panel. **This is an observation, not an established
mechanism.** Neither the cause (why runtime extents lead to that allocation) nor
the effect (that the drains cost the 93 µs) has been isolated. The
256 × 128 × 12288 × 128 bounded image (`n_*`) differs in bytes but not in this
pattern; its static image is the same HSACO as the rows case.

## Probe 3: the M-bounded rows, with ranges and attribution

Bounded − static, µs median per member (member 2 = gate product), and the
program ratio:

| M | gate product bounded [min, max] | static [min, max] | Δ gate | Δ other six members | program ratio |
|---|---|---|---|---|---|
| 1 | 818.0 [817.0, 818.2] | 736.6 [736.2, 737.0] | +81.5 | +1.5 (down product) | 1.101 |
| 2 | 818.5 [815.9, 819.4] | 736.8 [736.4, 737.2] | +81.7 | +1.2 | 1.101 |
| 8 | 819.7 [818.6, 820.5] | 738.2 [737.9, 738.7] | +81.5 | +1.4 | 1.101 |
| 16 | 930.8 [930.1, 931.8] | 930.8 [930.2, 932.2] | 0.0 | −0.3 | 1.000 |
| 32 | 931.0 [929.6, 932.2] | 931.3 [928.6, 933.0] | −0.3 | −0.4 | 0.999 |
| 64 | 931.3 [929.1, 934.1] | 930.1 [926.7, 931.3] | +1.2 | −0.2 | 1.002 |
| 128 | 1096.9 [1061.8, 1099.1] | 1004.2 [1000.7, 1006.5] | +92.7 | +0.8 | 1.083 |
| 256 | 1167.6 [1146.1, 1335.3] | 1074.2 [1060.6, 1191.4] | +93.4 | −0.4 | 1.103 |

**The M = 1 anomaly is real and is entirely in the gate product.** At M ≤ 8 the
bounded and static windows do not overlap, and the 81.5 µs gap is ≥ 97% of the
program gap. The ranges are tight, within 3.5 µs. The other members are equal
within 1.5 µs. The cost is not monotone in M: it is zero across 16–64 and returns at
128–256 (+93 µs). Static at M ≤ 8 is itself 194 µs faster than static at M = 16
with the same `[16, 32]` tile. The bounded image captures 113 µs of that saving.
At M = 256 the program gap (137 µs) is larger than the member sum (≈ 94 µs), and
both arms' windows reach 1191–1335 µs. Treat that row's program ratio as noisy.

## Files

`device-tests.txt`, `panel_min1.json`, `panel_min256.json`, `rows.json`,
`isa.json`, `isa/` (HSACO, disassembly, notes, backend IR), `probe.py`.
