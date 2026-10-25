# gfx1201 W8A8 small-grid 64x32 split

Owner: ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync: GFX1201-W8A8-SMALLGRID-64X32-20261025 (follows GFX1201-W8A8-SHORT-RAGGED-20261016).

## Change

The native PM Schedule rule `selectFp8W8A8BlockScalePanel` keeps the measured
four-wave LDS envelope from 2026-10-16 unchanged:

- gfx1201, NK weight, W8A8 E4M3, FP32 K128/N128 scales
- 64 < M < 128 with M not a multiple of 32
- N 1024..3072, a multiple of 128
- K 1024..2048

Inside that envelope, it now takes **64x32** tiles instead of 64x64 when the
64x64 grid has fewer workgroups than the RX 9070 XT's 64 CUs
(`ceil(M/64) * N/64 < 64`). Waves, staging and pipeline depth are unchanged.
The Python `blockscale_panel_oracle` mirrors the split, and
`lower_blockscale` still refuses a Schedule that the oracle does not
reproduce. Outside the envelope nothing changes. MXFP8/MXFP4, KN, other scale
contracts and gfx1151 are untouched; gfx1151 has no FP8 WMMA.

## Protocol

- Host: Tajasarus (RX 9070 XT, gfx1201), WSL2, ROCm 10.0, and its own
  assertions-ON LLVM/MLIR 23.1.1 build of `tessera-opt`.
- Revision: `source_commit` df9ebcbf5, clean tree. Its C++ is identical to
  38f9b64ae, the commit that carries the rule.
- Correctness before timing: every arm is checked against an fp64 oracle.
  F32 output uses a relative bound of 1e-5. BF16 output uses one bf16
  rounding (2^-7), the same bound AITER's bf16 store is held to.
- Timing: device-clock marker windows, paired and interleaved in one process,
  each cross-checked against HIP events. Every arm in every file is
  `admissible`.
- Comparison: AITER (Triton 3.8.0, BF16 output) is compared only with
  Tessera's BF16 arm.
- No hardware counters are available on WSL2 (no `/dev/kfd`).

## Result: `automatic-route-ab.json`

The production route (new rule) is compared with the previous geometry,
forced as an `lds:64x64:4:1` override in the same process. Times are device
medians in microseconds.

| M,N,K | 64x64 grid | Selected | BF16 old/new | BF16 new/AITER | F32 old/new |
|---|---:|---|---:|---:|---:|
| 65,1024,1024 | 32 | 64x32 | 1.218 | 0.917 | 1.160 |
| 97,1024,2048 | 32 | 64x32 | 1.272 | 0.916 | 1.240 |
| 125,1024,1024 | 32 | 64x32 | 1.291 | 0.929 | 1.240 |
| 81,1024,1536 | 32 | 64x32 | 1.244 | 0.992 | 1.173 |
| 71,1024,2048 | 32 | 64x32 | 1.211 | 0.877 | 1.187 |
| 100,1024,1024 | 32 | 64x32 | 1.254 | 0.895 | 1.196 |
| 65,1536,1024 | 48 | 64x32 | 1.134 | 0.848 | 1.116 |
| 97,1536,2048 | 48 | 64x32 | 1.059 | 0.899 | 1.049 |
| 113,1536,1536 | 48 | 64x32 | 1.142 | 0.948 | 1.093 |
| 127,2048,1024 | 64 | 64x64 | 1.000 | 0.992 | 0.995 |
| 127,2048,1536 | 64 | 64x64 | 1.000 | 1.106 | 1.008 |
| 97,2048,2048 | 64 | 64x64 | 1.006 | 0.947 | 1.012 |
| 75,2560,1536 | 80 | 64x64 | 0.999 | 0.954 | 1.009 |
| 113,3072,2048 | 96 | 64x64 | 1.003 | 0.867 | 1.009 |

- **Selected rows:** all nine improve, by 1.06–1.29x (BF16) and 1.05–1.24x
  (F32). Each now beats AITER, at 0.85–0.99x its time; at the old geometry
  they ran 0.95–1.23x.
- **Controls:** the five unchanged rows stay within ±1.2%.

## Where the boundary comes from

- **`sweep-candidates.json`** (diagnostic): seven LDS variants were tried on
  three panel rows. Only 64x32 with four waves at depth 1 beat AITER. Depth 2,
  128x32/w8, 32x64/w2 and 64x32/w2 were all slower.
- **`sweep-envelope.json`** (diagnostic): 64x64 vs 64x32 across ten shapes.
  - Below 64 workgroups, 64x32 won at every point.
  - At exactly 64, the result splits by K: 64x32 wins at K=1024/1536 and loses
    at K=2048.
  - From 80 workgroups up, 64x64 wins.

  The strict `<` is therefore the measured boundary. The 64-workgroup split by
  K is recorded, not tuned around.
- **`panel-main-129256f95.json`**: the W8A8 panel on `main` before this change.

## Open

- **127x2048x1536, 1.106x AITER:** at the 64-workgroup boundary, 64x32 measured
  1.06x in the diagnostic sweep. One shape is not a rule.
- **1024x3072x1536, 1.03x AITER (BF16):** a large-M row outside this envelope.
  It is unchanged here and remains open.
- **AITER is not yet beaten across the whole envelope.** No claim extends past
  the rows above.

## Tests

- Unit (Tajasarus): `tests/unit/test_rocm_fp8_blockscale.py` plus MXFP4-folded
  and WMMA-generated, 169 passed. The 15 skips are the gfx11-only 16x16x16
  WMMA fragment lanes.
  - The new `test_small_grid_splits_n_when_64x64_underfills_the_cus` checks
    both the oracle and the native Schedule.
- Device (Tajasarus, `TESSERA_ROCM_CHIP=gfx1201`,
  `TESSERA_GFX1201_DEVICE_PROOF=1`): `tests/device/rocm/test_fp8_blockscale_w8a8.py`,
  160 passed. The LDS-body cases are bit-equal to the fp64 oracle on exact
  inputs and bitwise equal to the register panel. Fixtures include two new
  64x32 cases and one 64x64 control.

File hashes: `sha256.txt`.
