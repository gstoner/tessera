# gfx1201 opted-in folded MXFP4 prefill

Tajasarus (RX 9070 XT, `gfx1201`) recorded this packet from source revision
`a9ce310f82296dfd0e7661322c7d6bf96bd13821`. The benchmark checked the
exact K32 output against an independent FP32-dequantized sample, required full
BF16 agreement among exact, folded, and pinned Radiance on a lossless-fold
payload, then alternated HIP-event timing across the three engines. The
folded package binds its E4M3 `[N,K]` payload and E8M0 `[N]` row reference to
a separate versioned ABI and records the fold's measured loss metadata.

| prefill M×N×K | exact K32 ms | folded BM256/TM4 ms | Radiance ms | exact / folded | folded / Radiance |
|---|---:|---:|---:|---:|---:|
| 256×5120×8704 | 0.506398 | 0.177464 | 0.142300 | 2.85× | 1.25× |
| 1024×17408×5120 | 4.358864 | 1.229560 | 0.889614 | 3.55× | 1.38× |

The folded HSACO emits 32 FP8 WMMAs, uses 138 VGPRs and 25,600 bytes of LDS,
and has no scratch or register spills. The policy is approximate even when
the benchmark's particular input folds without loss. Separate Tajasarus tests
exercise deliberate E4M3 underflow, ragged N, and refusal of a changed
load-time payload. The exact K32 route remains the correctness oracle and the
default execution route; the folded route requires explicit opt-in.

The remaining 24–38% gap to Radiance is measured, not closed. This version
stores expanded E4M3 weights at load time, doubling weight bytes versus packed
E2M1, and stages a BM256/TM4 tile with 16-byte vector copies. Further work
must compare physical weight traffic and A/B staging instruction counts before
changing the selector.

See [evidence.json](evidence.json) for samples, hashes, ISA, and resources.
