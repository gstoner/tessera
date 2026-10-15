# gfx1201 folded fragment read scheduling

Owner: ROCM-MXFP4-W4A8-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key: GFX1201-FOLDED-FRAGMENT-GROUPS-20261015.

## Native pipeline and selection

The existing explicitly approximate folded prefill Graph operation flows through
native Graph -> Schedule -> Tile -> ROCm Target -> ROCDL/LLVM -> HSACO and its
checked launch ABI. Tile-to-ROCm owns physical selection; Python only reports
the serialized Target decision and binds images. This does not promote folded
approximation to the default logical MXFP4 route.

The optional Target attribute fragment_read_schedule = "k32_groups" is selected
only for the unpacked rocm_mxfp4_w4a8_folded_prefill_v1 contract when:

- M is divisible by 256;
- N is at least 8192 and divisible by 64;
- K is at least 1024 and divisible by 64;
- row_guard is cta and staging_prefetch is register_next_slab.

Absence preserves the prior instruction schedule. Explicit wrong types, unknown
values, partial panel proofs and incompatible folded schedules are rejected.
Packed folded storage retains its separate contract. gfx1151, NVIDIA, Apple and
x86 receive no physical schedule change.

## Materialization and identity

Each K64 slab contains four K16 panels. The native materializer emits ROCDL
scheduling groups after panels 1 and 3: six DS reads and sixteen WMMA operations
per group. These are LLVM scheduling constraints, not additional GPU workgroup
barriers. The same typed LDS views, fragment accumulation order, scale handling,
output rounding and workgroup synchronization remain in place.

The Target attribute participates in native image identity. Projected runtime
M/N/K images preserve it and both whole-panel proofs; shape extents remain bound
by the checked package descriptor. Runtime-K images still require positive K64.
The selection thresholds are performance policy rather than a replacement for
buffer extents, layout checks or bounds guards. Different authored shapes may
reuse an image only within its remaining physical contract.

The row reference, including its existing zero-sentinel and reserved-code
behavior, is unchanged. Numerical equivalence compares identical folded inputs;
it does not claim that folding equals the original per-K-block source tensor.

## Evidence and limits

The original broad N>=4096 policy was rejected: M256/N4096/K1024 regressed
4.35 percent in both run orders, with regressions also measured at K2048/3072.
The narrowed bulk-N candidate improves the five selected cases in both orders:
roughly 1.5-4.1 percent in the first run and 1.9-3.7 percent in reverse order.
Captured graph dispatch plus kernel latency is separate from ordinary launch
windows and public host time. There are no WSL hardware counter measurements
and no fresh Radiance comparison.

The production packet and validation receipts are maintained under
benchmarks/baselines/gfx1201_folded_fragment_groups_20261015/.
Broader MXFP4 M256/Radiance attribution, W8A8 short/ragged coverage, other cache
families and general ROCm route closure remain open. No sibling parity is inferred.
