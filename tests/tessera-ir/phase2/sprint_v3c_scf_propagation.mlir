// RUN: tessera-opt --tessera-symdim-equality %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V3c (2026-05-22) — scf.for / scf.if region propagation.
//
// SymbolicDimEqualityPass now recurses into scf.for and scf.if body
// regions:
//   scf.for: iter_args inherit dim-names from init operands; the
//            scf.yield operands must match (loop must be
//            name-invariant) ⇒ SYMDIM_LOOP_YIELD_MISMATCH on conflict.
//   scf.if : both branches' yields must agree ⇒
//            SYMDIM_IF_BRANCH_MISMATCH on conflict.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: scf.for invariant — yield matches iter_args' dim-names. ✓
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @scf_for_invariant
func.func @scf_for_invariant(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]]
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // iter_arg inherits ["B", "D"] from %x; body yields the same
  // tensor (name-invariant); V3c silent.
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %x)
        -> tensor<4x8xf32> {
    scf.yield %acc : tensor<4x8xf32>
  }
  return %r : tensor<4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: scf.for yields a tensor whose propagated dim-names diverge
// from the iter_arg's names.  SYMDIM_LOOP_YIELD_MISMATCH.
//
// The body transposes the iter_arg ["B", "D"] → ["D", "B"] and yields
// that — breaks loop invariance.
// ─────────────────────────────────────────────────────────────────────────

func.func @scf_for_yield_mismatch(%x: tensor<4x4xf32>) -> tensor<4x4xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]]
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %x)
        -> tensor<4x4xf32> {
    %t = "tessera.transpose"(%acc) {
          tessera.dim_names_in = ["B", "D"],
          tessera.dim_names_out = ["D", "B"]
         } : (tensor<4x4xf32>) -> tensor<4x4xf32>
    // expected-error @+1 {{SYMDIM_LOOP_YIELD_MISMATCH: scf.for yield operand 0 dim-names disagree with the corresponding iter_arg's dim-names}}
    scf.yield %t : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: scf.if — both branches yield ["B", "D"].  Result dim-names
// well-defined; subsequent transpose silently cross-checks. ✓
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @scf_if_branches_agree
func.func @scf_if_branches_agree(
    %cond: i1, %x: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<8x4xf32>
    attributes {
      tessera.arg_dim_names = [[], ["B", "D"], ["B", "D"]]
    } {
  %r = scf.if %cond -> tensor<4x8xf32> {
    scf.yield %x : tensor<4x8xf32>
  } else {
    scf.yield %y : tensor<4x8xf32>
  }
  // %r now carries ["B", "D"]; transpose declares the same — silent.
  %z = "tessera.transpose"(%r) {
        tessera.dim_names_in = ["B", "D"],
        tessera.dim_names_out = ["D", "B"]
       } : (tensor<4x8xf32>) -> tensor<8x4xf32>
  return %z : tensor<8x4xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: scf.if — then-branch yields ["B", "D"], else-branch yields
// ["D", "B"] (via transpose).  SYMDIM_IF_BRANCH_MISMATCH on result 0.
// ─────────────────────────────────────────────────────────────────────────

func.func @scf_if_branches_disagree(
    %cond: i1, %x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32>
    attributes {
      tessera.arg_dim_names = [[], ["B", "D"], ["B", "D"]]
    } {
  // expected-error @+1 {{SYMDIM_IF_BRANCH_MISMATCH: scf.if result 0 has different dim-names in then-branch vs else-branch}}
  %r = scf.if %cond -> tensor<4x4xf32> {
    scf.yield %x : tensor<4x4xf32>
  } else {
    %t = "tessera.transpose"(%y) {
          tessera.dim_names_in = ["B", "D"],
          tessera.dim_names_out = ["D", "B"]
         } : (tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %t : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}
