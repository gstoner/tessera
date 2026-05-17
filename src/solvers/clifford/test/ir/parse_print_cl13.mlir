// RUN: ts-clifford-opt %s | FileCheck %s
//
// Parse + print round-trip on Cl(1,3) — the Minkowski-spacetime signature.
// Verifies the dialect carries the algebra attribute correctly across
// different signatures (v1 allow-list: Cl(3,0) and Cl(1,3)).

module {
  func.func @cl13_inner_is_mass_squared(
      %p : tensor<16xf32>) -> f32 {
    // ⟨p, p⟩ in Cl(1,3) computes the rest mass squared
    // m² = E² - |p|² — Lorentz invariant by signature.
    %m2 = "tessera_clifford.inner"(%p, %p)
        { algebra = [1, 3, 0], dtype = "fp32" }
        : (tensor<16xf32>, tensor<16xf32>) -> f32
    return %m2 : f32
  }
}

// CHECK: tessera_clifford.inner
// CHECK-SAME: algebra = [1, 3, 0]
