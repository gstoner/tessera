# gfx1201 MXFP4 production-schedule evidence

This packet records the first measured decode/prefill schedule split for
`ROCM-MXFP4-W4A8-1` on Tajasarus, an AMD Radeon RX 9070 XT (`gfx1201`). It is
bound to Tessera commit `a387f8c93c2850d42caf1df222e712178b0659dc` and to
the generator and benchmark hashes in `evidence.json`.

## Method

- Inputs are the same logical E4M3 activations, packed E2M1 weights, E8M0 K32
  scale plane, and per-token FP32 scale for every engine.
- E8M0 exponent deltas are at most two, so the independently implemented
  row-reference fold is exact for this comparison.
- Every engine must produce the same BF16 bits before it is timed.
- Three operand copies rotate between launches; production weight footprints
  exceed the device last-level cache.
- Timings are HIP events around 12 launches, with six warmups and nine samples.
- Effective weight bandwidth counts packed weights plus their E8M0 scale plane.

Radiance was independently built from
`ggz14/radiance-vllm-mxfp4@dfdfa3832922c9a4253133f09c1f5c0d39748fc7`.
libr4d was independently built from the canonical
`StillDeadcode/libr4d@5dc6302b87d598d1d3bf2ad3b50aab365461a63c` and is
reported only inside its `M <= 64` contract. No source was copied from either
tree; neither inspected root exposed a license or SPDX declaration.

## Results

| Workload (M x N x K) | Tessera schedule | Tessera ms | Radiance ms | libr4d ms |
|---|---:|---:|---:|---:|
| decode 8 x 5120 x 8704 | split-K 8 | 0.088602 | 0.032314 | 0.035013 |
| decode 8 x 17408 x 5120 | split-K 8 | 0.155802 | 0.085725 | 0.104815 |
| prefill 256 x 5120 x 8704 | group-M 8 | 0.633868 | 0.139085 | not applicable |
| prefill 1024 x 17408 x 5120 | group-M 8 | 4.845018 | 0.881319 | not applicable |

The new schedules pass the five-row scalar-oracle/WMMA/generic-materializer
owning-device fixture. They are 1.10–2.02x faster than the direct schedule in
the tuning sweeps, but Radiance remains 1.82–5.50x faster on these shapes.
This packet therefore proves schedule function and measured progress, not a
production selector promotion.

Remaining work is fragment/prepacked decode loading, multi-stage prefill,
cache-modifier and waves-per-EU tuning, then resource/ISA evidence for the
winner.
