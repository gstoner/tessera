// RUN: ts-clifford-opt --tessera-clifford-grade-fusion %s | FileCheck %s
//
// Attachment only.  That the restriction actually PRUNES is a separate claim
// with its own single-function fixture (input_grade_fusion_prunes.mlir): a
// count assertion cannot share a file with other functions that also expand.
//
// W1.4 — `input_grades`, the mirror of `output_grades`.
//
// `output_grades` prunes the Cayley table by WHICH RESULTS ARE WANTED.
// `input_grades` prunes it by WHICH INPUTS CAN BE NON-ZERO. Both are
// compile-time sparsity and only the first existed; the integrated plan cites
// `MultivectorSpec.grades` reaching no consumer as a live Decision #29
// violation, and this is its MLIR-side half.
//
// In Cl(3,0) a grade-1 operand has 3 non-zero blades out of 8, so
// `geo_product(grade(1,a), grade(1,b))` has 3x3 = 9 contributing table entries
// rather than 64 — and the result is grades {0, 2} without anyone writing an
// output restriction to say so.

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

    // Both operands are grade-restricted, so BOTH sides get annotated.
    // CHECK: tessera_clifford.geo_product
    // CHECK-SAME: tessera.clifford.input_grades_lhs = [1]
    // CHECK-SAME: tessera.clifford.input_grades_rhs = [1]
    %p = "tessera_clifford.geo_product"(%av, %bv)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %p : tensor<8xf32>
  }

  // Only ONE side restricted: the other must stay unannotated rather than
  // defaulting to "empty". An absent declaration means unrestricted, and
  // getting that backwards would silently emit a zero product.
  // CHECK-LABEL: func.func @one_side_only
  func.func @one_side_only(%a : tensor<8xf32>, %b : tensor<8xf32>)
      -> tensor<8xf32> {
    %av = "tessera_clifford.grade"(%a)
        { grades = [2], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK: tessera_clifford.geo_product
    // CHECK-SAME: tessera.clifford.input_grades_lhs = [2]
    // CHECK-NOT: input_grades_rhs
    %p = "tessera_clifford.geo_product"(%av, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %p : tensor<8xf32>
  }

  // Neither side restricted — the negative case. A pass that annotates
  // everything is not conservative, it is an unmeasured claim a later pass
  // will act on (Decision #10a: an eligibility-marking pass ships a fixture
  // whose correct output is NO annotation).
  // CHECK-LABEL: func.func @unrestricted_gets_nothing
  func.func @unrestricted_gets_nothing(%a : tensor<8xf32>, %b : tensor<8xf32>)
      -> tensor<8xf32> {
    // CHECK: tessera_clifford.geo_product
    // CHECK-NOT: input_grades
    %p = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %p : tensor<8xf32>
  }

}
