# gfx1201 W8A8 follow-up, 2026-09-28

Tajasarus (RX 9070 XT, `gfx1201`), Tessera `2a70e08a`, freshly rebuilt
assertions-enabled `tessera-opt` (LLVM/MLIR 23.1.1). The JSON files here are
unedited output of `benchmark_gfx1201_fp8_blockscale.py`, using the matched
LLVM 23.1.1 tools, AITER's unmodified gfx1201 `gemm_a8w8_blockscale`, five
paired device-clock windows, fp64 oracle checks and HIP-event/host-wall
cross-checks. `current_main.json` records the three original gap shapes.
The other packets sweep the production panel against opt-in LDS 64x128 and/or
128x64 variants. For example:

```bash
TESSERA_LLVM_BIN=$HOME/.local/share/tessera-toolchains/llvm-23.1.1/bin \
TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1 \
flock /tmp/tessera-timing.lock $HOME/wbs-aiter-venv/bin/python \
  benchmarks/rocm/benchmark_gfx1201_fp8_blockscale.py \
  --compiler build-assertions/tools/tessera-opt/tessera-opt \
  --aiter-root $HOME/programming/aiter \
  --shape 97,1024,1024 --shape 127,1024,1024 \
  --no-production --with-aiter --sweep prod:nk \
  --sweep lds:128x64:8:1:-1:-1:-1:nk --windows 5 \
  --output /tmp/tessera-gfx1201-w8a8-ragged-row-general-20260928.json
```

The existing Schedule chooses a one-wave register panel at M=97–127 because
those heights are not divisible by 32. The bounded 128x64 LDS candidate is
2.4–3.4x faster on the measured N=1024–24576, K=1024–4096 rows and agrees
with the fp64 oracle. Examples (microseconds, candidate / production):

| M×N×K | Production | LDS 128x64 | AITER |
| --- | ---: | ---: | ---: |
| 97×1024×1024 | 25.94 | 10.56 | 8.62 |
| 100×8192×1024 | 53.83 | 24.67 | 28.69 |
| 100×24576×1536 | 282.29 | 83.42 | 97.87 |
| 127×2048×4096 | 95.87 | 29.19 | 30.23 |

M=96 and M=128 keep their current panels: in the `boundary_sweep.json`,
the opt-in panels are slower than production at those boundaries. The
candidate is a Tile override of the old Schedule and supplies selection
evidence. After rebuilding the changed compiler on Tajasarus, the native
Schedule selected the LDS panel at M=97–127, and the focused unit and
device suites passed 117/117 and 22/22 respectively. `post_rule.json`
remeasures the production route with that compiler:

| M×N×K | New production (µs) | New production / AITER | Selection |
| --- | ---: | ---: | --- |
| 96×8192×1024 | 24.03 | 0.863 | unchanged register |
| 97×1024×1024 | 10.46 | 1.201 | new LDS |
| 100×8192×1024 | 25.01 | 0.873 | new LDS |
| 100×24576×1536 | 90.91 | 0.909 | new LDS |
| 127×2048×4096 | 29.77 | 0.973 | new LDS |
| 128×8192×1024 | 23.60 | 0.824 | unchanged LDS |
| 200×8192×1024 | 46.68 | 1.076 | unchanged LDS |

These are five paired device-clock windows with the unchanged AITER reference
and an fp64 oracle. The M=100,N=24576,K=1536 opt-in and production HSACO
digests match; their timing difference across packets is run variation.

The M=200, N=8192, K=1024 candidate repeats at 42.0–42.1 µs against
46.1–46.9 µs production (AITER 43.8–44.3 µs). M=300 and
M=200, N=24576, K=1536 regress with that candidate, so no broad M=200
selector change is made. The K=1536 large-M and MXFP4 M=256 investigations
remain open.

## Bounded short-K panel follow-up

The same 128x64 LDS panel was swept against the incumbent 128x128 body at
K=1024. `m200_neighbors.json`, `m200_envelope.json`, and
`m200_boundaries.json` cover M=192–300, N=4096–12288 and K=1024–2048;
`shortk_m200_stage_sweep.json` also varies the LDS K stage and prefetch.
Within M=192–255, whole-128 N=8192–10240, K=1024, the paired candidate won at every
measured corner. Examples (device-clock microseconds, candidate / incumbent):

| M×N×K | 128x64 | incumbent 128x128 | AITER |
| --- | ---: | ---: | ---: |
| 192×8192×1024 | 41.58 | 45.96 | 35.97 |
| 200×8192×1024 | 42.11 | 46.73 | 43.63 |
| 255×8192×1024 | 43.99 | 48.32 | 45.54 |
| 192×10240×1024 | 49.89 | 53.46 | 55.98 |
| 255×10240×1024 | 52.20 | 56.46 | 58.91 |

The selection now stays inside that envelope. At N=4096/6144, K=1536/2048
or M=300, 128x64 does not sustain a win. A K64 stage and prefetch were slower
on the original M=200 gap and the large-M K=1536 row. After rebuilding the
compiler, `m200_interior.json` compared the new production rule with an
explicit old-panel override at N=8320/9216/10112. All five rows were
neutral or faster, but 200x8320 won by only 0.38 µs; the selector requires
N to be a whole 128-column scale block and makes no ragged-N claim. Focused
selector/device cases passed 30/30 before the final N guard and the full W8A8
unit/device pair passed 184/184 afterward; ROCm lit passed 81 with one unsupported.
`m200_post_rule.json`
confirms that the production package selects 128x64 at M=192/200/255,
keeps the old panel outside the rule, and passes the fp64 oracle. The new
M=200,N=8192,K=1024 production kernel measured 44.16 µs against AITER's
44.42 µs in that run; its HSACO SHA-256 matches the pre-rule opt-in candidate.
The absolute timing varies across runs, so the interleaved pre-rule comparison
is the evidence for the panel choice. M=192 remains behind AITER, and the
large-M short-K and ragged K=1536 gaps remain open.
