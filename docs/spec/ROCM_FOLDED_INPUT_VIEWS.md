---
last_updated: 2026-10-16
---

# Folded gfx1201 input storage views

Owner: ROCM-MXFP4-W4A8-1 / LAYOUT-ALG-1.
Synchronization key: GFX1201-FOLDED-INPUT-VIEWS-20261016.

The explicit physical profile rocm_mxfp4_w4a8_folded_row_pitched_v1 preserves
the existing full-K FP32 reduction, E4M3 folded values, E8M0 row reference,
explicit approximate policy, wide-scale epilogue fallback and BF16 output.

## Semantic and storage boundary

This named contract belongs to four-operand tessera.scaled_matmul. A physical
scaled-contract tag on ordinary matmul is diagnosed before scale operands are
read, consistently with the shared ROCm/NVIDIA contract preflight.

The Graph operands are unencoded logical tensor<MxKxui8> and tensor<NxKxui8>.
Each is a standard MLIR tensor.extract_slice of a function-entry dense
unencoded storage argument, or its canonicalized identity entry argument: zero origin, unit steps, all storage rows. The parent row
width is LDA or LDB, at least K and divisible by 16. First delivery requires
static positive extents, M > 64, K divisible by 64. Other origins, steps,
partial rows, encoded tensor layouts and non-entry roots require a later view composition contract.

Native scheduling derives row pitches from the storage types and includes
them in schedule lineage. Tile bufferization extracts pointers from the
parent roots, retaining caller-owned input lifetimes. The result is freshly
allocated. No tensor is silently relabeled as contiguous.

## Checked ABI and image identity

The native ABI carries five flat expanded memrefs followed by i64
M, N, K, LDA, LDB. Flat memref sizes cover the physical dense roots:
M*LDA and N*LDB bytes. The logical K remains the reduction bound; only
global row addressing uses LDA/LDB. The distinct package ABI is:

    tessera.rocm.mxfp4_w4a8.a_bfold_sa_rowref_o_m_n_k_lda_ldb.e4m3_e4m3_e8m0_bf16.approx_bm256_tm4.v1

Package guards bind exact storage shapes and scalar pitches, prevent signed
64-bit span overflow, and retain the load-time hash of the logical weight
region. Padding is neither arithmetic input nor part of that logical hash.
Input roots remain alive through completion. The logical A/B, scale vectors,
result and physical A/B roots must be unencoded tensors. This dense ABI cannot
discard a conflicting or unknown tensor encoding; another physical layout
requires its own verified view/ABI contract. Reserved row-reference code
255 remains invalid.

Native image projection removes authored pitch values and carries
runtime_row_strides with zero pitch sentinels. One image can therefore
serve all four independent A/B padding combinations within the same schedule
class. Runtime dimensions, capacities and panel classes remain checked.

## Validation and performance policy

The bounded profile executes on RX 9070 XT / gfx1201. Numerical tests use an
independent decoded FP64 oracle, distinct E4M3 NaN padding, full/ragged panels
and identical image bytes across pitch combinations. Ordinary MLIR
canonicalization may remove identity slices; scheduling preserves those
entry roots and the same checked physical ABI. The canonicalized route has
separate owning-device numerical proof.

Seven-trial forward/reverse attribution compares dense/dense, padded-A/dense-B,
dense-A/padded-B and padded/padded under one image at ten M256 shapes. Raw
device-clock and HIP-event windows are separate from public launch wall time.
See the [frozen packet](../../benchmarks/baselines/gfx1201_folded_input_pitches_20261016/README.md).
No end-to-end performance improvement or default promotion is established.
The broad per-column cost and prepared/resident public route remain open.

This profile does not promote padding as a default optimization, establish a
cache mechanism, claim general view/AD support or transfer physical schedules
to gfx1151, NVIDIA, Apple or x86.
