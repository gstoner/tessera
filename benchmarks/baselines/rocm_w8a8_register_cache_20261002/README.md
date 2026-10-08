# gfx1201 register W8A8 native image reuse

Owner ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6-ROCM-MATMUL-CACHE.
Sync ROCM-W8A8-REGISTER-IMAGE-CACHE-2026-10-02.

This is revision-bound register-route evidence. Its LDS status is superseded
by the [runtime-M/N LDS packet](../rocm_w8a8_lds_mn_cache_20261002/README.md).
Original source/compiler hashes and measurements remain unchanged.

## Native contract

The compiler projects a verified one-wave global W8A8 Target directive to an
explicit runtime_shape contract with m/n/k=0. The typed native producer obtains
all addressing and loop bounds from the checked M/N/K launch ABI. Scale
semantics, macro-K, panel dimensions, storage/output, layout, raster order/group,
module attributes and physical options remain image keys. The shape-specific
descriptor retains buffer/scalar guards and Schedule/Graph/Tile ancestry.

This is Graph -> Schedule -> Tile -> checked ROCm Target -> native typed
fragments -> ROCDL/LLVM -> HSACO. Python binds compiler products and handles
package caching; it does not emit a kernel or translate Tile into Target.
The checked Target product cache is bounded and keyed by Tile text, compiler
binary and all pipeline options. Every package still validates the descriptor.

The Target consumer now also carries the scaled carrier's raster order/group.
Their earlier omission left grouped diagnostics executing row-major; this fix
preserves the requested physical schedule. The production selector remains
unchanged. LDS retains static dimensions: its grid/edge masks and K staging
still depend on them. Other scaled contracts are rejected by this projection.

## Exact-device proof

Tajasaurus, AMD Radeon RX 9070 XT, active gfx1201. The current compiler passed
240 native projection, W8A8 device and package checks. KN/NK and fp32/bf16
each reuse one image across (M,N,K)=(17,23,256), (31,45,512), (63,61,384),
while Schedule hashes and shape guards differ. Varying scale_n creates a cold,
distinct image and remains numerically correct. Invalid runtime N/K and scale
capacity are rejected without output changes. Grouped-M, grouped-N and
column-major diagnostic schedules match independent Tile-input numerical
execution and change physical image identity. Existing LDS coverage passes.
Host WSL package/diagnostic/pass checks: 415 passed.

The twelve benchmark arms passed the independent fp64 oracle before timing
and after each public runtime call. Source hashes match the authoritative
checkout; active GPU, compiler, image and entry identities are recorded.
The first shape is cold per contract; subsequent shapes are warm image hits.
Repeated construction uses the same verified Target and image product.

## Timing domains

First/repeated package time excludes Graph/Schedule/Tile frontend construction
(which is recorded separately) and device execution. Kernel time is the
compiler-built device-clock window, with HIP event and host cross-checks.
End-to-end time is an uninstrumented public launch including staging,
transfers, dispatch and completion; compilation is excluded.

| M,N,K | Layout/output | First package ms | Repeated package ms | Kernel us | Launch ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 17,23,256 | kn | 234.582 | 13.929 | 22.368 | 0.900 |
| 31,45,512 | kn | 40.500 | 13.462 | 123.965 | 1.147 |
| 63,61,384 | kn | 39.759 | 13.911 | 32.305 | 0.918 |
| 17,23,256 | nk | 164.543 | 13.682 | 20.717 | 0.956 |
| 31,45,512 | nk | 38.698 | 13.066 | 119.888 | 0.966 |
| 63,61,384 | nk | 39.086 | 13.594 | 27.973 | 0.954 |
| 17,23,256 | kn+bf16 | 169.202 | 13.869 | 22.046 | 0.973 |
| 31,45,512 | kn+bf16 | 37.700 | 13.137 | 123.428 | 1.100 |
| 63,61,384 | kn+bf16 | 38.430 | 13.884 | 32.691 | 0.980 |
| 17,23,256 | nk+bf16 | 165.809 | 13.694 | 20.831 | 0.961 |
| 31,45,512 | nk+bf16 | 38.793 | 13.576 | 123.506 | 1.067 |
| 63,61,384 | nk+bf16 | 38.327 | 14.142 | 28.747 | 0.898 |

These micro-shape timings are diagnostic. No AITER comparison, isolated
kernel speedup, selector promotion or broader shape performance claim.

## Remaining and sibling outcomes

LDS W8A8 shape-independent image identity and MXFP4/folded scaled keys remain
open. Broader W8A8 short/ragged-K performance obligations remain open.
gfx1151 has no RDNA4 FP8 WMMA instruction; no gfx1151 physical execution claim.
Apple, NVIDIA and x86 have no changed physical consumer or runtime ABI in
this ROCm-specific slice. Existing sibling architecture obligations remain open.
