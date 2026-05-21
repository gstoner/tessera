// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s

// Visual-complex-core — cross-lane (GA × EBM) IR-visible fixture.
//
// Proves the GA and EBM compiler surfaces co-exist in one module flow:
//
//   * tessera_clifford.rotor_sandwich    — GA lane
//   * tessera_clifford.grade_projection  — GA lane
//   * tessera_ebm.energy_quadratic       — EBM lane (Clifford-norm based)
//   * tessera_ebm.langevin_step          — EBM lane
//   * tessera_ebm.partition_exact        — EBM lane
//   * tessera_ebm.logsumexp              — EBM scalar invariant
//
// Both lanes share the same Cl(3, 0) layout (rank-2 tensor with last
// axis = 8 blades), so a refactor that breaks the GA-EBM bridge by
// changing the multivector layout surfaces as a lit failure here.

// CHECK-LABEL: func @visual_complex_block
// CHECK-DAG:   tessera_clifford.rotor_sandwich
// CHECK-DAG:   tessera_clifford.grade_projection
// CHECK-DAG:   tessera_ebm.energy_quadratic
// CHECK-DAG:   tessera_ebm.langevin_step
// CHECK-DAG:   tessera_ebm.partition_exact
// CHECK-DAG:   tessera_ebm.logsumexp
// CHECK-DAG:   algebra_signature = [3, 0, 0]
// CHECK-DAG:   grade = 2 : i64
// CHECK-DAG:   eta = 5.000000e-02 : f32
func.func @visual_complex_block(
    %x: tensor<4x8xf32>,
    %target: tensor<8xf32>,
    %rotor: tensor<4x8xf32>,
    %seed: i64
) -> (tensor<4x8xf32>, tensor<4x8xf32>, f32) {

  // ── GA lane: sandwich application ──────────────────────────────────
  %sandwiched = "tessera_clifford.rotor_sandwich"(%rotor, %x) {
      algebra_signature = [3, 0, 0]
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

  // ── EBM lane: Clifford-norm-based quadratic energy ────────────────
  %energies_init = "tessera_ebm.energy_quadratic"(%sandwiched, %target) :
      (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4xf32>

  // ── EBM lane: one annealed Langevin step ──────────────────────────
  %y = "tessera_ebm.langevin_step"(%sandwiched, %target, %seed) {
      eta = 5.000000e-02 : f32,
      temperature = 1.000000e-01 : f32
  } : (tensor<4x8xf32>, tensor<8xf32>, i64) -> tensor<4x8xf32>

  // ── GA lane: grade-2 projection of the post-anneal state ──────────
  %bivec = "tessera_clifford.grade_projection"(%y) {
      algebra_signature = [3, 0, 0],
      grade = 2 : i64
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>

  // ── EBM lane: final partition function ────────────────────────────
  %energies_final = "tessera_ebm.energy_quadratic"(%y, %target) :
      (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4xf32>
  %Z = "tessera_ebm.partition_exact"(%energies_final) {
      temperature = 1.000000e-01 : f32
  } : (tensor<4xf32>) -> f32

  // Generic logsumexp cross-check (the op_catalog's
  // `tessera.logsumexp` is the real Graph IR primitive; the generic
  // form keeps this fixture self-contained).
  %log_z = "tessera_ebm.logsumexp"(%energies_final) :
      (tensor<4xf32>) -> f32

  return %y, %bivec, %Z : tensor<4x8xf32>, tensor<4x8xf32>, f32
}
