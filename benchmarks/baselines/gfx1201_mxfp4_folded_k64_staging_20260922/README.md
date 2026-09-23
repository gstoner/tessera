# GFX1201 folded MXFP4 K64 staging

Owner `ROCM-MXFP4-W4A8-1`; sync `GFX1201-FOLDED-K64-STAGING-2026-09-22`.
The selected RX 9070 XT on Tajasarus ran paired, alternating HIP-event trials
with the exact K32 Tessera route and pinned Radiance TN2/WPERM1 comparator.
Lossless-fold inputs produced bitwise-identical BF16 outputs; an independent
FP32-dequantized sample checked the exact route. The new production K64
specialization removes only the redundant `kb + off < K` predicates from the
A/B vector-copy path. K32-tail packages retain guarded copies.

The first pre-promotion ablation measured 0.1660 versus 0.1775 ms on
`256x5120x8704`, and 1.1532 versus 1.2338 ms on `1024x17408x5120`.
Production-specialized reruns measured 0.1660 versus Radiance 0.1409 ms and
1.1695 versus Radiance 0.8914 ms. These are kernel HIP-event medians, not
end-to-end model or measured DRAM times. Sample variation makes single-percent
effects inconclusive.

Unconditionally issuing the four K16 compute steps raised VGPRs from 109 to
117 and regressed against the K64-only specialization, so that lever is
refused. Separate address-hoist and scheduling-barrier probes were also small
or mixed. A full-wave epilogue fast path helped the smaller shape but raised
VGPRs to 171; it is not selected. The remaining expanded-E4M3 versus packed
E2M1 weight-traffic difference is a hypothesis for the residual gap, not a
measured DRAM attribution. A packed-weight execution ABI would need its own
Graph physical contract, Schedule/Tile/Target carrier, exact numerical oracle,
layout identity, and independent device proof; it cannot silently replace the
explicit approximate folded representation.

`evidence.json` is the current-revision exact-device paired packet. Historical
packets remain pinned to their original generator hashes, not re-labeled as
evidence for this code revision.
