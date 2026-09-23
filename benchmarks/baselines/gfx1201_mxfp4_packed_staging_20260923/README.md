# GFX1201 packed-fragment producer staging ablation

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-STAGING-ABLATION-2026-09-23`.

Run `benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py` with
`--include-batched` to interleave the exact K32, expanded folded, packed
table, packed integer, B-only batching, A-only batching, A+B batching,
paired-K16 scale reuse, and pinned Radiance routes. Each shape must pass
the independent sampled exact oracle and bitwise BF16 agreement before HIP
event timing. Package/model-load work is outside the timed kernel. The
packet binds each selected-symbol ISA digest to its timed HSACO payload.

These are opt-in schedule ablations, not automatic selector candidates.
The versioned Graph physical layout and native ABI are unchanged. The
baseline v1 packet remains an immutable historical control; the v2 packet
records the current revision and every ablation on the exact gfx1201 host.
