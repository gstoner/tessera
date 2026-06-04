// Stage 16B — registered GA/EBM Graph IR ops with verifier-hardened metadata.
//
// RUN: %tessera_strict_opt %s | FileCheck %s

// CHECK-LABEL: func.func @ebm_controls
// CHECK: tessera.ebm.inner_step
// CHECK: tessera.ebm.refinement
// CHECK: tessera.ebm.langevin_step_philox
func.func @ebm_controls(%y: tensor<2x4xf32>, %g: tensor<2x4xf32>,
                        %n: tensor<2x4xf32>, %seed: tensor<1xi64>,
                        %ctr: tensor<4xi64>) -> tensor<2x4xf32> {
  %0 = tessera.ebm.inner_step %y, %g
      {eta = 5.000000e-02 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = tessera.ebm.refinement %0, %g, %n
      {eta = 5.000000e-02 : f64, steps = 4 : i64,
       temperature = 1.000000e+00 : f64, noise_scale = 2.000000e-01 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>)
        -> tensor<2x4xf32>
  %2 = tessera.ebm.langevin_step_philox %1, %g, %seed, %ctr
      {eta = 5.000000e-02 : f64, temperature = 1.000000e+00 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi64>, tensor<4xi64>)
        -> tensor<2x4xf32>
  return %2 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @ebm_candidate_ops
// CHECK: tessera.ebm.decode_init
// CHECK: tessera.ebm.self_verify
// CHECK: tessera.ebm.partition_exact
func.func @ebm_candidate_ops(%x: tensor<2x4xf32>,
                             %noise: tensor<2x3x4xf32>,
                             %e: tensor<2x3xf32>)
    -> (tensor<2x4xf32>, tensor<f32>) {
  %c = tessera.ebm.decode_init %x, %noise
      {operandSegmentSizes = array<i32: 1, 0, 1>,
       steps = 3 : i64, strategy = "noise"}
      : (tensor<2x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %best = tessera.ebm.self_verify %e, %c
      {reduction = "hard_argmin"}
      : (tensor<2x3xf32>, tensor<2x3x4xf32>) -> tensor<2x4xf32>
  %z = tessera.ebm.partition_exact %e
      {temperature = 1.000000e+00 : f64, reduction = "logsumexp"}
      : (tensor<2x3xf32>) -> tensor<f32>
  return %best, %z : tensor<2x4xf32>, tensor<f32>
}

// CHECK-LABEL: func.func @ebm_manifold_ops
// CHECK: tessera.ebm.bivector_langevin_step
// CHECK: tessera.ebm.sphere_langevin_step
func.func @ebm_manifold_ops(%s: tensor<2x8xf32>, %g: tensor<2x8xf32>,
                            %n: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %b = tessera.ebm.bivector_langevin_step %s, %g, %n
      {eta = 1.000000e-02 : f64, grade = 2 : i64,
       manifold = "bivector", projection = "grade",
       temperature = 1.000000e+00 : f64}
      : (tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>)
        -> tensor<2x8xf32>
  %o = tessera.ebm.sphere_langevin_step %b, %g
      {eta = 1.000000e-02 : f64, manifold = "sphere",
       normalized_state = true, projection = "tangent"}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %o : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @clifford_ops
// CHECK: tessera.clifford.geometric_product
// CHECK: tessera.clifford.outer_product
// CHECK: tessera.clifford.inner_product
// CHECK: tessera.clifford.reverse
// CHECK: tessera.clifford.grade_project
// CHECK: tessera.clifford.norm
// CHECK: tessera.clifford.rotor_sandwich
func.func @clifford_ops(%a: tensor<2x8xf32>, %b: tensor<2x8xf32>)
    -> (tensor<2x8xf32>, tensor<2xf32>) {
  %gp = tessera.clifford.geometric_product %a, %b
      {p = 3 : i64, q = 0 : i64, coefficient_layout = "blade_last",
       signature = [1 : i64, 1 : i64, 1 : i64]}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %op = tessera.clifford.outer_product %a, %b
      {p = 3 : i64, q = 0 : i64}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %ip = tessera.clifford.inner_product %a, %b
      {p = 3 : i64, q = 0 : i64}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %r = tessera.clifford.reverse %gp {p = 3 : i64, q = 0 : i64}
      : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %pr = tessera.clifford.grade_project %r
      {p = 3 : i64, q = 0 : i64, grade_mask = [1 : i64, 2 : i64]}
      : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %norm = tessera.clifford.norm %pr {p = 3 : i64, q = 0 : i64}
      : (tensor<2x8xf32>) -> tensor<2xf32>
  %rot = tessera.clifford.rotor_sandwich %pr, %a
      {p = 3 : i64, q = 0 : i64, grade_mask = [1 : i64]}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %rot, %norm : tensor<2x8xf32>, tensor<2xf32>
}
