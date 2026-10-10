# Native NVFP4 ingest physical leaf — gfx1201

Owner ROCM-NVFP4-INGEST-1. Sync ROCM-NATIVE-INGEST-LEAF-2026-10-03.
The full five-slice goal remains active. No strategy/default promotion.

## Implemented boundary

The registered ROCm Target operation tessera_rocm.nvfp4_requantize declares
NVFP4 packed E2M1 rows, E4M3 K16 scale bytes and one f64 global scale per
projection. Strict row boundaries preserve independent merged gate/up globals.
The native quantization materializer builds a GPU MLIR kernel, lowered through
ROCDL/LLVM to HSACO. No Python HIP source/template performs conversion.

One thread owns a K32 destination block. It decodes source values, selects the
code-energy-weighted exponent seed, then searches seed ±4 for the joint minimum
decoded-weight SSE. Candidate ties choose closest seed then lower exponent;
E2M1 midpoint ties choose the lower magnitude as in the declared ingest oracle.
Deterministic f64 eight-accumulator reductions match the contiguous K32 oracle.
Packed codes, group-major E8M0 bytes and per-block signal/error pairs are outputs.
The statistics support measured loss/SQNR without repeating Python conversion.

The operation verifier rejects wrong architectures, layout/policy conflicts,
K misalignment, incomplete/duplicate projection boundaries and extent overflow.
The materializer also checks the module's architecture before emitting a kernel.
Existing f64 arithmetic instruction availability was checked against the RDNA4
JSON ISA archive.

**This is a physical leaf, not completed frontend integration.**
Graph/Schedule/Tile ownership, checked public package ABI and the public ingest
call still need integration. Tests and the benchmark use a private six-memref
resident launcher with independent device allocations and completion before
freeing buffers. They do not establish a public @jit route or package replay.

## Exact-device evidence

Tajasaurus / AMD Radeon RX 9070 XT / live gfx1201; matching rebuilt compiler.
focused.txt: 17 passes (ten native verifier/materializer contract tests and
seven exact-device numerical cases). Device cases include all E2M1 codes,
zero scales, zero/signed-zero code blocks, raw finite nonnegative E4M3 bytes,
non-power-of-two globals, ragged thread grids and unequal merged projection
globals. Destination codes and exponents match the existing joint-SSE oracle
bitwise; signal/error match independently decoded f64 arithmetic.

gfx1201.json records three synthetic merged-projection benchmark cases,
source/compiler/image/Target identity and three resident event windows.
Correctness is checked after timing. Each window contains ten launches:

| Merged N,K | Resident conversion ms | Python oracle wall ms |
| --- | ---: | ---: |
| 67,256 | 0.163762 | 3.662 |
| 513,1024 | 0.370814 | 82.703 |
| 4097,4096 | 7.038248 | 2606.914 |

These timing scopes differ. Device events include dispatch and omit transfers,
module loading and host metadata; oracle wall includes CPU conversion and
error evaluation. No end-to-end speedup claim follows. Inputs are synthetic,
not pinned checkpoint proof. The old quantization lane's regression file has
one compiler test passing and nine legacy runtime-visibility skips; those
skips are not exact-device validation.

## Remaining engineering

Add target-neutral semantic Graph conversion with explicit loss policy,
content-addressed Schedule ownership, Tile buffers/lifetimes and the checked
six-buffer ABI. Preserve projection globals, layout and numeric policy through
every boundary. Then execute pinned checkpoint conversion followed by native
packed MXFP4 matmul, with separate conversion/consumer/end-to-end timing.
FP8, MXFP8 and MXFP4 remain independent mandatory quality/correctness/performance
gates before final/default strategy decisions. Wider W1.1 and AD integration,
attention envelopes and ROCm route/performance obligations remain active.
