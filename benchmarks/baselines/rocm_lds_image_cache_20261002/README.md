# Native LDS matmul image reuse

Owner E2E-REAL-6-ROCM-MATMUL-CACHE; sync ROCM-LDS-IMAGE-CACHE-2026-10-02.

The scheduled Graph/Tile artifact passes through native Tile-to-ROCm lowering
and MLIR physical identity projection. The projected portable Target directive
selects the existing typed LDS producer; no Python kernel construction was
added. Native compilation keys retain staging, wave geometry and other
pipeline configuration. The generator rejects nonportable LDS directives.

Exact Radeon 8060S/gfx1151 and RX 9070 XT/gfx1201 tests cover fp16/bf16,
plain and bias+ReLU, three shapes including ragged M/N/K, and a numerical
2x2-to-1x2 wave-recipe change. Shape changes hit the image cache; wave changes
produce a distinct cold image. ROCm regression suite: gfx1151 127 passed/
12 skipped; gfx1201 129 passed/10 skipped. Shared package, recorder, diagnostic
and pass metadata gates on Super-Bear: 423 passed/49 unavailable cases skipped.

Each architecture has static and bounded dynamic fused fp16 six-shape packets.
Correctness is checked before timing, during warmup and after each end-to-end
sample with absolute tolerance 2e-4. Packets carry live device identity, source
and compiler hashes, compile states and module-cache counters. HIP-event
kernel samples and runtime.launch end-to-end samples are separate;
end-to-end includes allocation, transfers, dispatch and completion.
These measurements characterize the current LDS route. No speedup or default
schedule promotion is claimed. Split-K and scaled image-key closure remain open.

## Subsequent identity correction

The split-partition slice preserves Schedule k_blocks in Target IR. These
packets predate that correction and remain historical; some gfx1201 shapes
selected different macro-K recipes which the earlier adapter dropped. Current
image reuse is proved within a fixed physical recipe, with distinct images
for differing macro-K counts. See ../rocm_split_partition_cache_20261002/README.md.
