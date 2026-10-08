# ROCm K-unrolled register matmul image reuse

Owner E2E-REAL-6-ROCM-MATMUL-CACHE; synchronization key
ROCM-K-UNROLL-IMAGE-CACHE-2026-10-02.

The native MLIR kernel identity projection now preserves K-unroll configuration
through Tile-to-Target compilation and native image generation. Unscaled
fp16/bf16 register-staged, split-K=1 packages can reuse an image across shapes
with K-unroll=2. Changing the unroll recipe produces a cold, distinct image.
LDS, split-K, scaled matmul, and wider dtype envelopes remain open.

Exact-device tests on Radeon 8060S (gfx1151) and RX 9070 XT (gfx1201) prove
three shapes, including ragged M/N/K, for fp16/bf16 with and without bias+ReLU:
four owning-device cases passed per host (four sibling cases skipped).
Focused identity/cache/lifetime regression suites: 109 passed/7 skipped on
gfx1151, 111 passed/5 skipped on gfx1201. Shared scheduled-consumer suite:
124 passed/49 unavailable backend cases skipped on Super-Bear.

Each static/fused benchmark packet contains six fp16 shapes, one cold compile
and five warm image reuses, numerical checks before timing and after each
end-to-end sample, live device identity, compiler/source hashes, and module
cache counters. HIP events measure kernel launch work; runtime.launch timing
includes allocation, transfers, dispatch and synchronization. Compilation and
Schedule construction have separate timings. Aggregate kernel medians across
different shapes are diagnostic only; no speedup or route promotion is claimed.

The initial broad AMD run also selected AVX-512 tests whose runtime was absent
from these ROCm compiler builds. That run is not a passing shared hardware gate.
The corrected shared suite was run on the configured Super-Bear compiler host;
AVX-512 device proof remains separate from this architecture-specific change.

Bounded dynamic M/N/K plus bias+ReLU was also checked and timed independently
on both devices (dynamic_fused.json): all six active shapes share one image
and retain descriptor capacity guards. This extends the measured envelope;
the dynamic packets have the same correctness and timing-domain checks.

## Subsequent identity correction

The split-partition slice preserves Schedule k_blocks in Target IR. These
packets predate that correction and remain historical; some gfx1201 shapes
selected different macro-K recipes which the earlier adapter dropped. Current
image reuse is proved within a fixed physical recipe, with distinct images
for differing macro-K counts. See ../rocm_split_partition_cache_20261002/README.md.
