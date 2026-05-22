// RUN: tessera-opt --tessera-symdim-equality %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V5 (2026-05-22) — SymbolicDimEqualityPass V1.
//
// Closes the 4th MLIR-verifier gap in SHAPE_SYSTEM.md §11.2.
// V1 dispatches on tessera.reshape / tessera.transpose / tessera.matmul.
// Today the dialect registers `transpose` and `matmul` but NOT
// `reshape` — V1's lit coverage exercises the two registered ops,
// and the pass keeps the reshape branch for when the op is added
// (a separate V2 task tracked in SHAPE_SYSTEM §11.2).
//
// Stable diagnostic codes the lit fixture matches on:
//   SYMDIM_BINDING_VIOLATION              — function-level equation broken
//   SYMDIM_MATMUL_CONTRACT_VIOLATION      — matmul lhs/rhs symbol mismatch
//
// V1 scope: 1 positive + 2 negative cases.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: D = H * Dh holds with H=4, Dh=16, D=64.  A transpose op
// with consistent in/out dim_names (a true permutation) passes the
// per-op contract.  Pass should succeed silently.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @symdim_transpose_ok
// CHECK:       tessera.transpose
func.func @symdim_transpose_ok(%x: tensor<2x4x64xf32>) -> tensor<4x2x64xf32>
    attributes {
      tessera.dim_bindings = ["D = H * Dh"],
      tessera.dim_sizes = { H = 4 : i64, Dh = 16 : i64, D = 64 : i64,
                            B = 2 : i64, T = 4 : i64 }
    } {
  %y = "tessera.transpose"(%x) {
    tessera.dim_names_in = ["B", "T", "D"],
    tessera.dim_names_out = ["T", "B", "D"]
  } : (tensor<2x4x64xf32>) -> tensor<4x2x64xf32>
  return %y : tensor<4x2x64xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE 1: SYMDIM_BINDING_VIOLATION
// dim_bindings declares D = H * Dh; dim_sizes claims H=4, Dh=16, D=65.
// 4 * 16 = 64 ≠ 65 ⇒ verifier rejects with the stable diagnostic.
// ─────────────────────────────────────────────────────────────────────────

// expected-error @+1 {{SYMDIM_BINDING_VIOLATION: binding 'D = H * Dh' violated: D = 65 but product of RHS = 64}}
func.func @symdim_binding_broken(%x: tensor<2x4x65xf32>) -> tensor<2x4x65xf32>
    attributes {
      tessera.dim_bindings = ["D = H * Dh"],
      tessera.dim_sizes = { H = 4 : i64, Dh = 16 : i64, D = 65 : i64 }
    } {
  return %x : tensor<2x4x65xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE 2: SYMDIM_MATMUL_CONTRACT_VIOLATION
// dim_names_lhs = [M, K] and dim_names_rhs = [J, N] — the matmul
// contracts on K (lhs.back()) against J (rhs.front()).  K ≠ J ⇒
// verifier rejects.
// ─────────────────────────────────────────────────────────────────────────

func.func @symdim_matmul_contract_broken(
    %A: tensor<4x8xbf16>, %B: tensor<8x16xbf16>) -> tensor<4x16xf32>
    attributes {
      tessera.dim_sizes = { M = 4 : i64, K = 8 : i64, J = 8 : i64, N = 16 : i64 }
    } {
  // expected-error @+1 {{SYMDIM_MATMUL_CONTRACT_VIOLATION: lhs contracts on 'K' but rhs contracts on 'J'}}
  %C = "tessera.matmul"(%A, %B) {
    tessera.dim_names_lhs = ["M", "K"],
    tessera.dim_names_rhs = ["J", "N"],
    transposeA = false,
    transposeB = false
  } : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xf32>
  return %C : tensor<4x16xf32>
}
