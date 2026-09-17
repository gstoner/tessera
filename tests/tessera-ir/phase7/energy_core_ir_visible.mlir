// RUN: tessera-opt %s | FileCheck %s
// REQUIRES: tessera-ebm

// Energy-core IR-visible fixture.
//
// Pins the compiler-facing substrate an EBM library needs, in the dialect's
// own spellings:
//   * tessera_ebm.energy       — E(context, y) through a Graph IR energy fn
//   * tessera_ebm.langevin_step — y' = y − η ∂E + √(2ηT) ξ
//   * tessera_ebm.partition_z  — the partition function
//
// Rewritten 2026-09-16: this fixture predated the EBM dialect and used
// placeholder names (`energy_quadratic`, `annealing_schedule`,
// `partition_exact`, `logsumexp`) that `tessera_ebm` never declared, plus an
// f32-attributed `langevin_step` that the real op rejects. It parsed only
// because --allow-unregistered-dialect covered the prefix; registering the
// dialect in tessera-opt made every one of those a hard error, since MLIR
// refuses an unknown op — or a malformed known one — inside a REGISTERED
// dialect whatever that flag says. An annealing schedule is genuinely absent
// from the dialect and is not faked here; the temperature is a step attribute.

// CHECK-LABEL: func @energy_core_block
// CHECK-DAG:   tessera_ebm.energy
// CHECK-DAG:   tessera_ebm.langevin_step
// CHECK-DAG:   tessera_ebm.partition_z
// CHECK-DAG:   temperature 1.000000e-01
// CHECK-DAG:   eta 5.000000e-02
// CHECK-DAG:   manifold "euclidean"
func.func @energy_core_block(
    %x_init: tensor<4x8xf32>,
    %target: tensor<4x8xf32>,
    %key: tensor<2xi64>
) -> (tensor<4xf32>, tensor<4xf32>, f32) {

  // Per-row energy of the initial state through the named energy function.
  %energies_init = "tessera_ebm.energy"(%target, %x_init) { energy_fn = @quadratic_energy } :
      (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4xf32>

  // One Langevin step at the coldest temperature of the anneal.
  %y_next, %key_next = "tessera_ebm.langevin_step"(%x_init, %key, %target) { operandSegmentSizes = array<i32: 1, 1, 0, 1>,
      energy_fn = @quadratic_energy,
      eta = 5.000000e-02 : f64,
      temperature = 1.000000e-01 : f64,
      manifold = "euclidean"
  } : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>)

  // Energies after the step.
  %energies_post = "tessera_ebm.energy"(%target, %y_next) { energy_fn = @quadratic_energy } :
      (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4xf32>

  // The partition function over the same energy.
  %Z, %diag = "tessera_ebm.partition_z"() {
      energy_fn = @quadratic_energy, method = "exact"
  } : () -> (f32, tensor<2xf32>)

  return %energies_init, %energies_post, %Z : tensor<4xf32>, tensor<4xf32>, f32
}

func.func private @quadratic_energy(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32>
