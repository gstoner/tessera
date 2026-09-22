# gfx1201 opted-in folded MXFP4 prefill

Tajasarus (RX 9070 XT, `gfx1201`) recorded this packet from source revision
`4368c825b866c3eb7b40761c5c17bbcb817e2665`. The benchmark checked the
exact K32 output against an independent FP32-dequantized sample, required full
BF16 agreement among exact, folded, and pinned Radiance on a lossless-fold
payload, then alternated HIP-event timing across the three engines. The
folded package binds its E4M3 `[N,K]` payload and E8M0 `[N]` row reference to
a separate versioned ABI and records the fold's measured loss metadata.

| prefill M×N×K | exact K32 ms | folded BM256/TM4 ms | Radiance ms | exact / folded | folded / Radiance |
|---|---:|---:|---:|---:|---:|
| 256×5120×8704 | 0.517027 | 0.181150 | 0.142784 | 2.85× | 1.27× |
| 1024×17408×5120 | 4.365548 | 1.239120 | 0.887218 | 3.52× | 1.40× |

The folded HSACO emits 32 FP8 WMMAs, uses 109 VGPRs and 25,600 bytes of LDS,
and has no scratch or register spills. The policy is approximate even when
the benchmark's particular input folds without loss. Separate Tajasarus tests
exercise deliberate E4M3 underflow, ragged N, and refusal of a changed
load-time payload. The follow-on regression proves that a lossless row exponent
of 254 combined with activation scale 2^-127 produces finite output 32 instead
of overflowing an intermediate. Two further cases preserve zero accumulators
and finite products when the combined FP32 scale overflows. All six device
tests pass. The exact K32 route remains the correctness oracle and the
default execution route; the folded route requires explicit opt-in.

The remaining 27–40% gap to Radiance is measured, not closed. This version
stores expanded E4M3 weights at load time, doubling weight bytes versus packed
E2M1, and stages a BM256/TM4 tile with 16-byte vector copies. Further work
must compare physical weight traffic and A/B staging instruction counts before
changing the selector.

See [evidence.json](evidence.json) for samples, hashes, ISA, and resources.
