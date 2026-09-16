// RUN: ts-clifford-opt --tessera-clifford-expand-product-table=expand-rotor-sandwich=true %s | FileCheck %s
//
// The whole product family lowers through the one compile-time table
// (2026-09-16): no tessera_clifford op survives, batched or not.

module {
  // Wedge keeps only disjoint-blade terms: e1 ^ e1 vanishes, so with both
  // inputs pure vectors the scalar coefficient is never accumulated.
  // CHECK-LABEL: func.func @wedge_batched
  // CHECK: scf.for
  // CHECK: arith.mulf
  // CHECK-NOT: tessera_clifford.
  func.func @wedge_batched(%a : tensor<16x8xf32>, %b : tensor<16x8xf32>) -> tensor<16x8xf32> {
    %r = "tessera_clifford.wedge"(%a, %b) { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    return %r : tensor<16x8xf32>
  }

  // CHECK-LABEL: func.func @left_contract_single
  // CHECK: tensor.from_elements
  // CHECK-NOT: tessera_clifford.
  func.func @left_contract_single(%a : tensor<8xf32>, %b : tensor<8xf32>) -> tensor<8xf32> {
    %r = "tessera_clifford.left_contract"(%a, %b) { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }

  // Scalar forms are typed [..., 1]: one scalar per multivector.
  // CHECK-LABEL: func.func @inner_and_norm
  // CHECK: tensor.empty() : tensor<4x1xf32>
  // CHECK: arith.maximumf
  // CHECK: math.sqrt
  // CHECK-NOT: tessera_clifford.
  func.func @inner_and_norm(%a : tensor<4x8xf32>, %b : tensor<4x8xf32>) -> (tensor<4x1xf32>, tensor<4x1xf32>) {
    %i = "tessera_clifford.inner"(%a, %b) { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x1xf32>
    %n = "tessera_clifford.norm"(%a) { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<4x8xf32>) -> tensor<4x1xf32>
    return %i, %n : tensor<4x1xf32>, tensor<4x1xf32>
  }

  // Unary maps are sign/permutation maps: negf only, no multiplies.
  // CHECK-LABEL: func.func @involutions
  // CHECK: arith.negf
  // CHECK-NOT: arith.mulf
  // CHECK-NOT: tessera_clifford.
  func.func @involutions(%a : tensor<3x8xf32>) -> (tensor<3x8xf32>, tensor<3x8xf32>, tensor<3x8xf32>, tensor<3x8xf32>, tensor<3x8xf32>) {
    %r = "tessera_clifford.reverse"(%a) { algebra = [3, 0, 0], dtype = "fp32" } : (tensor<3x8xf32>) -> tensor<3x8xf32>
    %g = "tessera_clifford.grade_involute"(%a) { algebra = [3, 0, 0], dtype = "fp32" } : (tensor<3x8xf32>) -> tensor<3x8xf32>
    %c = "tessera_clifford.conjugate"(%a) { algebra = [3, 0, 0], dtype = "fp32" } : (tensor<3x8xf32>) -> tensor<3x8xf32>
    %h = "tessera_clifford.hodge_star"(%a) { algebra = [3, 0, 0], dtype = "fp32" } : (tensor<3x8xf32>) -> tensor<3x8xf32>
    %p = "tessera_clifford.grade"(%a) { grades = [1, 3], algebra = [3, 0, 0], dtype = "fp32" } : (tensor<3x8xf32>) -> tensor<3x8xf32>
    return %r, %g, %c, %h, %p : tensor<3x8xf32>, tensor<3x8xf32>, tensor<3x8xf32>, tensor<3x8xf32>, tensor<3x8xf32>
  }

  // With expand-rotor-sandwich, rotor_sandwich expands to gp(gp(R, x),
  // reverse(R)) and lowers in the same run; without it the fused marker
  // survives for backends with a sandwich kernel (full_pipeline_cl30.mlir).
  // CHECK-LABEL: func.func @sandwich
  // CHECK: scf.for
  // CHECK-NOT: tessera_clifford.
  func.func @sandwich(%R : tensor<5x8xf32>, %x : tensor<5x8xf32>) -> tensor<5x8xf32> {
    %y = "tessera_clifford.rotor_sandwich"(%R, %x) { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<5x8xf32>, tensor<5x8xf32>) -> tensor<5x8xf32>
    return %y : tensor<5x8xf32>
  }
}
