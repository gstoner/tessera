// RUN: tessera-opt %s | FileCheck %s
// REQUIRES: tessera-ebm

// Visual-complex-core — cross-lane (GA x EBM) IR-visible fixture.
//
// Proves the GA and EBM compiler surfaces co-exist in one module flow, in
// both dialects' own spellings:
//
//   * tessera_clifford.rotor_sandwich — GA lane
//   * tessera_clifford.grade          — GA lane
//   * tessera_ebm.energy              — EBM lane
//   * tessera_ebm.langevin_step       — EBM lane
//   * tessera_ebm.partition_z         — EBM lane
//
// Both lanes share the same Cl(3, 0) layout (rank-2 tensor with last axis =
// 8 blades), so a refactor that breaks the GA-EBM bridge by changing the
// multivector layout surfaces as a lit failure here.
//
// Rewritten 2026-09-16 for the same reason as its two siblings: the fixture
// predated both dialects and used placeholder op names and f32 attributes
// that the registered dialects reject.

// CHECK-LABEL: func @visual_complex_block
// CHECK-DAG:   tessera_clifford.rotor_sandwich
// CHECK-DAG:   tessera_clifford.grade
// CHECK-DAG:   tessera_ebm.energy
// CHECK-DAG:   tessera_ebm.langevin_step
// CHECK-DAG:   tessera_ebm.partition_z
// CHECK-DAG:   algebra [3, 0, 0]
// CHECK-DAG:   grades [2]
// CHECK-DAG:   eta 5.000000e-02
func.func @visual_complex_block(
    %x: tensor<4x8xf32>,
    %target: tensor<4x8xf32>,
    %rotor: tensor<4x8xf32>,
    %key: tensor<2xi64>
) -> (tensor<4x8xf32>, tensor<4x8xf32>, f32) {

  // GA lane: sandwich application.
  %sandwiched = "tessera_clifford.rotor_sandwich"(%rotor, %x) {
      algebra = [3, 0, 0], dtype = "f32"
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

  // EBM lane: the energy of the rotated state.
  %energies_init = "tessera_ebm.energy"(%target, %sandwiched) { energy_fn = @quadratic_energy } :
      (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4xf32>

  // EBM lane: one annealed Langevin step.
  %y, %key_next = "tessera_ebm.langevin_step"(%sandwiched, %key, %target) { operandSegmentSizes = array<i32: 1, 1, 0, 1>,
      energy_fn = @quadratic_energy,
      eta = 5.000000e-02 : f64,
      temperature = 1.000000e-01 : f64,
      manifold = "euclidean"
  } : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>)

  // GA lane: grade-2 projection of the post-anneal state.
  %bivec = "tessera_clifford.grade"(%y) {
      grades = [2], algebra = [3, 0, 0], dtype = "f32"
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>

  // EBM lane: the partition function over the same energy.
  %Z, %diag = "tessera_ebm.partition_z"() {
      energy_fn = @quadratic_energy, method = "exact"
  } : () -> (f32, tensor<2xf32>)

  return %y, %bivec, %Z : tensor<4x8xf32>, tensor<4x8xf32>, f32
}

func.func private @quadratic_energy(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32>
