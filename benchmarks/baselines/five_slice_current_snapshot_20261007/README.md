# Current five-slice owning-host revalidation

Owner ROCM-NVFP4-INGEST-1 / W1.1 / E2E-REAL-6.
Sync FIVE-SLICE-CURRENT-SNAPSHOT-2026-10-07.

The authoritative source remains the unpublished next-five compiler branch.
This packet validates current source execution; it is not full completion
or a fresh full-suite/performance certificate.

Current LLVM/MLIR 23.1.1 native compiler SHA256:
e4e33848bc4b1b9378789dc2c47539a1f1c405ac0c93ec0e2df9713fdfe071d6.

## NVIDIA RTX 5070 / sm_120

The current owning-host lane passes 171 tests covering tensor program contracts,
public saved-LSE attention forward, public reverse and multi-result O/LSE VJP.
The lane combines host contract tests and exact-device tests; 171 is not a
count of independent GPU numerical profiles.
The earlier architecture-specific benchmark packets remain bound to their
recorded compilers/images. Their timings were not regenerated in this lane.

## ROCm RX 9070 XT / gfx1201

Current Python/runtime/tests are synchronized into an isolated scratch root.
Native leaf ingest, public conversion, frontend resident program and portable
replay gates initially report 47 passes and one environment failure:
the native image loader had no library path. The complete movement owner and
native image bridge are separate dylibs; movement discovery does not establish
image bridge discovery.

With TESSERA_ROCM_NATIVE_IMAGE_LIB bound to the verified owning-host library,
the affected fresh-process compiler-free replay case passes.
Both the original failed log and successful repair are retained.
Compiler/movement/image byte identities are recorded separately.
This receipt establishes the named static routes, not model quality or
dynamic/layout/composed AD acceptance.

## Delivery boundary

Origin is current; four existing unpublished commits alone contain the older
host conversion slice. A faithful PR must include the subsequent native
Graph/Schedule/Tile converter, checked resident owner and public frontend
integration. The later aggregate has substantial source and raw evidence;
it needs reviewable delivery slices, not an unqualified closure declaration.

Generic scaled-matmul batching/transpose closure, wider W1.1 composition,
ROCm performance obligations, final generated-doc/graph gates, fresh full-suite
green and publication remain open. Do not weaken coverage states or transfer
physical proof across architectures to make those gates appear complete.

## Subsequent bounded-row ingest envelope — 2026-10-09

The dependent GFX1201-BOUNDED-NVFP4-ROWS-20261009 slice now proves native MLIR
capacity export through Schedule/Tile/Target packaging and ordinary/portable
changing-row replay on RX 9070 XT. Eight new device cases, 36 existing static
cases and a final source-bound twelve-row timing packet establish that named
envelope. HIP capacity allocations remain fixed while active M changes.
See ../gfx1201_bounded_nvfp4_rows_20261009/README.md for timing domains and limits.

This is subsequent evidence, not a replacement for this historical snapshot.
Generic scaled batching/transpose, whole-model ingest quality, broader W1.1
composition, composed attention and ROCm performance obligations remain open.
