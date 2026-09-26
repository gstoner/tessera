// RUN: %trop %s -split-input-file -verify-diagnostics
// RUN: not %trop --generate-wmma-gemm-kernel %s --split-input-file 2>&1 | FileCheck %s --check-prefix=DIRECTIVE
//
// ROCM-SPLIT-K-1 at the Target IR boundary. `tessera_rocm.wmma_gemm` carries
// the split and its reduction order as a semantic pair (Decision #21a): the
// verifier refuses half a pair or any order but `ordered`. A well-formed split
// directive is accepted by the verifier but REFUSED by the generator's
// directive adapter, which has no split body (split-K is emitted only from the
// typed tile.matmul_kernel route) -- it must not be answered unsplit.

// expected-error @+1 {{ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split_k > 1 requires split_k_reduction = "ordered"}}
tessera_rocm.wmma_gemm {name = "a", m = 16 : i64, n = 16 : i64, k = 16 : i64, split_k = 2 : i64}

// -----

// expected-error @+1 {{ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split_k > 1 requires split_k_reduction = "ordered"}}
tessera_rocm.wmma_gemm {name = "b", m = 16 : i64, n = 16 : i64, k = 16 : i64, split_k = 2 : i64, split_k_reduction = "atomic"}

// -----

// expected-error @+1 {{ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split_k_reduction requires split_k > 1}}
tessera_rocm.wmma_gemm {name = "c", m = 16 : i64, n = 16 : i64, k = 16 : i64, split_k_reduction = "ordered"}

// -----

// expected-error @+1 {{ROCM_WMMA_GEMM_SPLIT_K_BAD_CONTRACT: split_k must be >= 1}}
tessera_rocm.wmma_gemm {name = "d", m = 16 : i64, n = 16 : i64, k = 16 : i64, split_k = 0 : i64}

// -----

// A consistent pair verifies; the directive adapter then refuses it (DIRECTIVE).
tessera_rocm.wmma_gemm {name = "e", m = 16 : i64, n = 16 : i64, k = 16 : i64, split_k = 2 : i64, split_k_reduction = "ordered"}

// DIRECTIVE: ROCM_SPLIT_K_{{UNSUPPORTED}}: the tessera_rocm.wmma_gemm directive adapter has no split-K body
