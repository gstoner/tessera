// RUN: ts-clifford-opt %s | FileCheck %s
//
// Parse + print round-trip for every Clifford op on a Cl(3,0) signature.
// No passes — just verify the dialect parses and prints cleanly.

module {
  func.func @cl30_full_surface(
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> tensor<8xf32> {

    // ── Core multivector ops (GA3) ────────────────────────────────────
    %gp = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>

    %gradek = "tessera_clifford.grade"(%gp)
        { grades = [2], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>

    %we = "tessera_clifford.wedge"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>

    %lc = "tessera_clifford.left_contract"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>

    %sc = "tessera_clifford.inner"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> f32

    %nr = "tessera_clifford.norm"(%a)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> f32

    %rv = "tessera_clifford.reverse"(%a)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>

    %gi = "tessera_clifford.grade_involute"(%a)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>

    %cj = "tessera_clifford.conjugate"(%a)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>

    %hs = "tessera_clifford.hodge_star"(%a)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>

    %ex = "tessera_clifford.exp"(%a)
        { algebra = [3, 0, 0], dtype = "fp32", terms = 24 }
        : (tensor<8xf32>) -> tensor<8xf32>

    %lg = "tessera_clifford.log"(%a)
        { algebra = [3, 0, 0], dtype = "fp32", terms = 64 }
        : (tensor<8xf32>) -> tensor<8xf32>

    %rs = "tessera_clifford.rotor_sandwich"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>

    return %rs : tensor<8xf32>
  }
}

// CHECK: tessera_clifford.geo_product
// CHECK: tessera_clifford.grade
// CHECK: tessera_clifford.wedge
// CHECK: tessera_clifford.left_contract
// CHECK: tessera_clifford.inner
// CHECK: tessera_clifford.norm
// CHECK: tessera_clifford.reverse
// CHECK: tessera_clifford.grade_involute
// CHECK: tessera_clifford.conjugate
// CHECK: tessera_clifford.hodge_star
// CHECK: tessera_clifford.exp
// CHECK: tessera_clifford.log
// CHECK: tessera_clifford.rotor_sandwich
