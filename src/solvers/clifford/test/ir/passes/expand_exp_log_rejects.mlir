// RUN: not ts-clifford-opt --tessera-clifford-expand-product-table %s 2>&1 | FileCheck %s
//
// exp admits an operand the pass can *prove* is a pure bivector, because the
// reference switches to a 24-term power series otherwise and which branch
// applies is a property of the value, not of the type. Guessing would make the
// compiled program and the reference disagree on the same input, so the pass
// refuses and says how to fix it.

// The greedy rewriter does not fix the order the two refusals are reported in.
// CHECK-DAG: exp admits an operand that is provably a pure bivector
// CHECK-DAG: tessera_clifford.grade keeping grade 2 only
func.func @exp_of_an_unproven_operand(%x : tensor<8xf32>) -> tensor<8xf32> {
  %e = "tessera_clifford.exp"(%x) { algebra = [3, 0, 0], dtype = "fp32" }
      : (tensor<8xf32>) -> tensor<8xf32>
  return %e : tensor<8xf32>
}

// The closed forms are Cl(3, 0)'s; no other signature has one here, and the
// series fallback is not emitted for any signature.
// CHECK-DAG: closed-form log is defined for Cl(3, 0) only, not Cl(2, 0, 0)
func.func @log_in_another_algebra(%x : tensor<4xf32>) -> tensor<4xf32> {
  %l = "tessera_clifford.log"(%x) { algebra = [2, 0, 0], dtype = "fp32" }
      : (tensor<4xf32>) -> tensor<4xf32>
  return %l : tensor<4xf32>
}
