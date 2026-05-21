// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s

// Generic Clifford-core IR-visible fixture.
//
// Pins the compiler-facing substrate a GA library needs for batched
// multivector workloads:
//   * tessera_clifford.rotor_from_axis   — rotor construction
//   * tessera_clifford.rotor_sandwich    — R x R†
//   * tessera_clifford.geometric_product — full Clifford product
//   * tessera_clifford.grade_projection  — pick a single grade
//   * tessera_clifford.norm_squared      — scalar invariant
//
// The shapes use Cl(3, 0)'s 8-blade multivector layout: ``tensor<Bx8xf32>``.
// Each op is in generic form so the fixture roundtrips cleanly without
// a registered Clifford dialect; a real lowering pass picks them up by
// name when shipping.

// CHECK-LABEL: func @clifford_core_block
// CHECK-DAG:   tessera_clifford.rotor_from_axis
// CHECK-DAG:   tessera_clifford.rotor_sandwich
// CHECK-DAG:   tessera_clifford.geometric_product
// CHECK-DAG:   tessera_clifford.grade_projection
// CHECK-DAG:   tessera_clifford.norm_squared
// CHECK-DAG:   algebra_signature = [3, 0, 0]
// CHECK-DAG:   grade = 2 : i64
func.func @clifford_core_block(
    %x: tensor<8x8xf32>,
    %bivec_axis: tensor<8x8xf32>,
    %angle: f32
) -> (tensor<8x8xf32>, tensor<8xf32>) {

  // Step 1: construct a rotor from an unnormalized bivector axis + angle.
  %r0 = "tessera_clifford.rotor_from_axis"(%bivec_axis, %angle) {
      algebra_signature = [3, 0, 0]
  } : (tensor<8x8xf32>, f32) -> tensor<8x8xf32>

  // Step 2: sandwich-apply the rotor to the input multivectors.
  %sandwich = "tessera_clifford.rotor_sandwich"(%r0, %x) {
      algebra_signature = [3, 0, 0]
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>

  // Step 3: geometric-product composition — rotor · x.
  %composed = "tessera_clifford.geometric_product"(%r0, %x) {
      algebra_signature = [3, 0, 0]
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>

  // Step 4: grade-2 projection of the sandwiched result.
  %bivec_part = "tessera_clifford.grade_projection"(%sandwich) {
      algebra_signature = [3, 0, 0],
      grade = 2 : i64
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  // Step 5: scalar invariant via norm-squared.
  %nsq = "tessera_clifford.norm_squared"(%composed) {
      algebra_signature = [3, 0, 0]
  } : (tensor<8x8xf32>) -> tensor<8xf32>

  return %composed, %nsq : tensor<8x8xf32>, tensor<8xf32>
}
