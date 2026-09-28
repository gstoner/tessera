# gfx1201 foundation batch 3 deduplicated device evidence

Owner: ROCM-FP8-BLOCKSCALE-1 and ROCM-MXFP4-W4A8-1. Sync:
`FOUNDATION-BATCH-3-DEDUP-2026-09-28`. Tajasarus RX 9070 XT, gfx1201,
WSL2, HIP 7.15, LLVM/MLIR 23.1.1. The source is the post-#875 branch
`codex/rocm-perf-batch3-dedup` at `0a2d7fa7a`; the reference compiler was
rebuilt from merged #875 (`48142ae8e`). Both are full HIP builds.

## W8A8

`w8a8_paired.json` is the paired, interleaved device-clock record from
`benchmark_gfx1201_fp8_blockscale.py`: seven windows of at least 5 ms,
with HIP events and wall time as witnesses. All twelve arms were admitted.
Every Tessera output matched the fp64 block-scale oracle before timing.
AITER came from the unchanged `f0966b0ae` checkout with Triton 3.8.0.
Numbers are median microseconds per launch; lower is faster.

| M x N x K | #875 | this branch | AITER |
|---|---:|---:|---:|
| 200 x 8192 x 1024 | 43.65 | 41.01 | 44.49 |
| 200 x 2048 x 2048 | 24.14 | 21.41 | 23.73 |
| 1024 x 3072 x 1536 | 73.01 | 71.76 | 70.30 |
| 1024 x 3072 x 1024 | 54.06 | 51.49 | 60.62 |

The two old M=200 regressions are below AITER on these exact rows. The
K=1536 row is 2.1% above AITER; this four-row sample does not close the whole
short-K or ragged-K envelope. `tests/device/rocm/test_fp8_blockscale_w8a8.py`
passed 57/57 and the two changed lit fixtures passed against this build.
The uniform scale load now selects scale block zero for masked columns,
preserving the generic fragment contract.

## Folded MXFP4 at M=256

`mxfp4_slope.json` uses one input copy, three fresh processes and seven
trials per shape. Radiance is revision `dfdfa383`, fragment-order weights,
with binary SHA-256 `c9f91bc8...`; no counters are exposed on this WSL2 host.
The selected Tessera route matched the independent exact K32 sampled
reference before timing. The diagnostics intentionally change output and
are attribution probes, never candidate kernels. Ratios below are medians
of device-clock time divided by the selected route; lower is faster.

| N (K=5120) | Radiance / selected | source control | no A stage | no A fetch | no A LDS write |
|---|---:|---:|---:|---:|---:|
| 8192 | 0.860 | 0.975 | 0.586 | 0.948 | 0.816 |
| 12288 | 0.835 | 0.983 | 0.597 | 0.931 | 0.948 |
| 16384 | 0.792 | 0.964 | 0.703 | 0.922 | 0.911 |

A fetch plus LDS restaging together dominate the removable cost at N=8192
and 12288. Removing either half alone saves much less, so these probes do
not isolate which half Radiance implements more cheaply. The source control
varies by 2-4%, which bounds smaller effects. At N=16384, the no-A gain is
smaller; the weight stream may contribute, but no traffic counters prove it.
The selected folded route remains opt-in. The numerical folded device suite
passed 32/32 on this compiler; no selector or default-route change is claimed.
