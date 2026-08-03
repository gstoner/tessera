// RUN: ts-clifford-opt --tessera-clifford-grade-fusion \
// RUN:   --tessera-clifford-expand-product-table %s | FileCheck %s
//
// W1.4 — the restriction must actually PRUNE, and the count is the proof.
//
// A declaration with no consumer reads as a closed contract while carrying
// nothing (Decision #29), so asserting that `input_grades` appears is not
// enough: the emitted arithmetic has to shrink.
//
// Measured in Cl(3,0): an unrestricted geo_product expands to 64 `arith.mulf`.
// With both operands declared grade 1 — 3 non-zero blades out of 8 — exactly
// 3x3 = 9 survive.
//
// ONE function on purpose: a counted directive followed by a negative one
// counts over the WHOLE input, so a second function that also expands would
// make the assertion meaningless -- as it did on the first attempt, where this
// lived alongside two other products and silently counted 97.
//
// (Directive names are spelled out only in the checks below; writing them in
// prose makes FileCheck parse the comment as a directive.)

module {
  // CHECK-LABEL: func.func @vector_times_vector
  func.func @vector_times_vector(%a : tensor<8xf32>, %b : tensor<8xf32>)
      -> tensor<8xf32> {
    %av = "tessera_clifford.grade"(%a)
        { grades = [1], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    %bv = "tessera_clifford.grade"(%b)
        { grades = [1], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    %p = "tessera_clifford.geo_product"(%av, %bv)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %p : tensor<8xf32>
  }
  // CHECK-COUNT-9: arith.mulf
  // CHECK-NOT: arith.mulf
}
