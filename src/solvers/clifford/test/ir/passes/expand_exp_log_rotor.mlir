// RUN: ts-clifford-opt --tessera-clifford-expand-product-table %s | FileCheck %s
//
// The closed forms on Cl(3, 0): exp of a bivector, log of a multivector, and the
// rotor constructor. All three become arith/math over the Cayley table with no
// Clifford op left, and the power series the numpy reference falls back to is
// never emitted (it would be 24 geometric products per multivector).
//
// exp needs cos and sin of a runtime |B|; log needs atan2; rotor_from_axis needs
// neither, because its angle is an attribute and both transcendentals fold to
// constants at compile time — one reciprocal and a scale is the whole kernel.

// CHECK-LABEL: func @sample_on_the_group
// CHECK-NOT:     tessera_clifford
// CHECK-DAG:     math.sqrt
// CHECK-DAG:     math.cos
// CHECK-DAG:     math.sin
// CHECK-NOT:     tessera_clifford
func.func @sample_on_the_group(%tangent : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %b = "tessera_clifford.grade"(%tangent) { grades = [2], algebra = [3, 0, 0], dtype = "fp32" }
      : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %r = "tessera_clifford.exp"(%b) { algebra = [3, 0, 0], dtype = "fp32" }
      : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// CHECK-LABEL: func @back_to_the_algebra
// CHECK-NOT:     tessera_clifford
// CHECK-DAG:     math.atan2
// CHECK-NOT:     tessera_clifford
func.func @back_to_the_algebra(%rotor : tensor<8xf32>) -> tensor<8xf32> {
  %l = "tessera_clifford.log"(%rotor) { algebra = [3, 0, 0], dtype = "fp32" }
      : (tensor<8xf32>) -> tensor<8xf32>
  return %l : tensor<8xf32>
}

// The constructor folds its transcendentals: no math.cos / math.sin survives,
// only the division by the axis magnitude.
// CHECK-LABEL: func @rotor_about_an_axis
// CHECK-NOT:     tessera_clifford
// CHECK-NOT:     math.cos
// CHECK-NOT:     math.sin
// CHECK-DAG:     math.sqrt
// CHECK-DAG:     arith.divf
func.func @rotor_about_an_axis(%axis : tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera_clifford.rotor_from_axis"(%axis)
      { algebra = [3, 0, 0], dtype = "fp32", angle = 0.78539816339744828 : f64 }
      : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
