// RUN: tessera-opt --tessera-symdim-equality %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V2-flow (2026-05-22) — SymbolicDimEqualityPass SSA-value
// dim-name propagation.  Extends V5 / V6a / V6b: instead of relying
// on the user to annotate EVERY op with `tessera.dim_names_*`, V2
// seeds dim-names from `tessera.arg_dim_names` on the function and
// propagates through transpose / matmul / reshape ops.
//
// Stable diagnostic code added by V2-flow:
//   SYMDIM_FLOW_INCONSISTENCY — propagated dim-names disagree with
//                                explicit per-op annotation.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: arg dim-names propagate through matmul without any per-op
// annotations.  The pass should succeed silently.  The matmul shapes
// must satisfy the ODS MatmulOp::verify too — it runs at parse time
// before the symdim pass.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @flow_propagation_ok
// CHECK:       tessera.matmul
func.func @flow_propagation_ok(
    %x: tensor<4x8xf32>, %w: tensor<8x16xf32>
) -> tensor<4x16xf32>
    attributes {
      tessera.arg_dim_names = [["M", "K"], ["K", "N"]]
    } {
  // V2 propagates: %x has dim-names [M, K], %w has [K, N].
  // Matmul output (no transpose) = lhs[:-1] + rhs[-1:] = [M, N].
  // No per-op annotations needed; V2-flow infers everything.
  %out = "tessera.matmul"(%x, %w) {transposeA = false, transposeB = false}
       : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: SYMDIM_FLOW_INCONSISTENCY (lhs annotation contradicts arg)
// User declares tessera.dim_names_lhs = [P, K] on a matmul, but the
// propagated dim-names from the function arg are [M, K].  V2 catches
// the P ≠ M disagreement.  Note: we keep K in both so the V1 matmul-
// contract check passes (it only sees the contracting symbol K=K);
// V2-flow is the only check that catches P ≠ M at position 0.
// ─────────────────────────────────────────────────────────────────────────

func.func @flow_lhs_inconsistency(
    %x: tensor<4x8xf32>, %w: tensor<8x16xf32>
) -> tensor<4x16xf32>
    attributes {
      tessera.arg_dim_names = [["M", "K"], ["K", "N"]]
    } {
  // expected-error @+1 {{SYMDIM_FLOW_INCONSISTENCY: propagated dim-names disagree with explicit 'tessera.dim_names_lhs' annotation}}
  %out = "tessera.matmul"(%x, %w) {
    transposeA = false, transposeB = false,
    tessera.dim_names_lhs = ["P", "K"],
    tessera.dim_names_rhs = ["K", "N"]
  } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE (backward-compat): no tessera.arg_dim_names → V2-flow
// falls through cleanly (V5/V6a/V6b functions keep working).
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @flow_no_arg_names_falls_through
// CHECK:       tessera.matmul
func.func @flow_no_arg_names_falls_through(
    %x: tensor<4x8xf32>, %w: tensor<8x16xf32>
) -> tensor<4x16xf32> {
  %out = "tessera.matmul"(%x, %w) {transposeA = false, transposeB = false}
       : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}
