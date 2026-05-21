// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s

// Generic Energy-core IR-visible fixture.
//
// Pins the compiler-facing substrate every EBM library needs:
//   * tessera_ebm.energy_quadratic         — E(x | y) = ½ ||x − y||²
//   * tessera_ebm.annealing_schedule       — T_max → T_min linear
//   * tessera_ebm.langevin_step            — y' = y − η ∂E + √(2ηT) ξ
//   * tessera_ebm.partition_exact          — Z via stable logsumexp
//   * tessera_ebm.logsumexp                — generic-form stable logsumexp
//
// The tessera_ebm.* ops are generic-form placeholders for the EBM lane's
// future first-class dialect.  ``tessera.logsumexp`` itself is in
// ``op_catalog.py`` as a Graph IR primitive — the lane benchmark exercises
// it through the Python frontend (see EnergyCoreModel) while keeping the
// lit fixture self-contained.

// CHECK-LABEL: func @energy_core_block
// CHECK-DAG:   tessera_ebm.energy_quadratic
// CHECK-DAG:   tessera_ebm.annealing_schedule
// CHECK-DAG:   tessera_ebm.langevin_step
// CHECK-DAG:   tessera_ebm.partition_exact
// CHECK-DAG:   tessera_ebm.logsumexp
// CHECK-DAG:   temperature = 1.000000e-01 : f32
// CHECK-DAG:   eta = 5.000000e-02 : f32
func.func @energy_core_block(
    %x_init: tensor<4x8xf32>,
    %target: tensor<8xf32>,
    %seed: i64
) -> (tensor<4x8xf32>, f32) {

  // Step 1: per-row quadratic energy E_i = ½||x_i - target||².
  %energies_init = "tessera_ebm.energy_quadratic"(%x_init, %target) :
      (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4xf32>

  // Step 2: build a 4-step linear annealing schedule.
  %sched = "tessera_ebm.annealing_schedule"() {
      n_steps = 4 : i64,
      T_max = 1.000000e+00 : f32,
      T_min = 1.000000e-01 : f32
  } : () -> tensor<4xf32>

  // Step 3: one Langevin step at the coldest temperature.
  %y_next = "tessera_ebm.langevin_step"(%x_init, %target, %seed) {
      eta = 5.000000e-02 : f32,
      temperature = 1.000000e-01 : f32
  } : (tensor<4x8xf32>, tensor<8xf32>, i64) -> tensor<4x8xf32>

  // Step 4: re-evaluate energies after the step.
  %energies_post = "tessera_ebm.energy_quadratic"(%y_next, %target) :
      (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4xf32>

  // Step 5: partition function via the dedicated EBM op…
  %Z_ebm = "tessera_ebm.partition_exact"(%energies_post) {
      temperature = 1.000000e-01 : f32
  } : (tensor<4xf32>) -> f32

  // …cross-checked against a generic logsumexp op surface.  The real
  // tessera.logsumexp lives in op_catalog as a Graph IR op (used at
  // runtime via Python ts.ops.logsumexp); the generic form here keeps
  // the lit fixture roundtripping cleanly without depending on the
  // op's custom syntax.
  %log_z_generic = "tessera_ebm.logsumexp"(%energies_post) :
      (tensor<4xf32>) -> f32

  return %y_next, %Z_ebm : tensor<4x8xf32>, f32
}
