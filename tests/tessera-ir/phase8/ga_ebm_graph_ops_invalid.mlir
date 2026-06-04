// Stage 16B — GA/EBM Graph IR verifier diagnostics.
//
// RUN: %tessera_strict_opt %s --verify-diagnostics -o /dev/null

func.func @ebm_temperature_without_noise(%y: tensor<2x4xf32>,
                                         %g: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{temperature > 0 requires a noise operand or RNG state}}
  %0 = tessera.ebm.langevin_step %y, %g
      {eta = 5.000000e-02 : f64, temperature = 1.000000e+00 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @ebm_noise_scale_without_noise(%y: tensor<2x4xf32>,
                                         %g: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{noise_scale > 0 requires a noise operand}}
  %0 = tessera.ebm.inner_step %y, %g
      {eta = 5.000000e-02 : f64, noise_scale = 1.000000e-01 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @ebm_bad_philox_counter(%y: tensor<2x4xf32>, %g: tensor<2x4xf32>,
                                  %seed: tensor<1xi64>, %ctr: tensor<3xi64>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{Philox counter tensor must have length 4}}
  %0 = tessera.ebm.langevin_step_philox %y, %g, %seed, %ctr
      {eta = 5.000000e-02 : f64, temperature = 1.000000e+00 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi64>, tensor<3xi64>)
        -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @ebm_decode_noise_needs_source(%x: tensor<2x4xf32>)
    -> tensor<2x3x4xf32> {
  // expected-error @+1 {{strategy="noise" requires noise operand or seed attr}}
  %0 = tessera.ebm.decode_init %x
      {operandSegmentSizes = array<i32: 1, 0, 0>,
       steps = 3 : i64, strategy = "noise"}
      : (tensor<2x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

func.func @ebm_self_verify_softmin_needs_temperature(
    %e: tensor<2x3xf32>, %c: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{reduction="softmin" requires temperature > 0}}
  %0 = tessera.ebm.self_verify %e, %c {reduction = "softmin"}
      : (tensor<2x3xf32>, tensor<2x3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @ebm_bivector_bad_grade(%s: tensor<2x8xf32>,
                                  %g: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  // expected-error @+1 {{bivector Langevin requires grade = 2}}
  %0 = tessera.ebm.bivector_langevin_step %s, %g
      {eta = 1.000000e-02 : f64, grade = 1 : i64}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

func.func @ebm_sphere_requires_normalized(%s: tensor<2x8xf32>,
                                          %g: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  // expected-error @+1 {{sphere Langevin requires normalized_state = true}}
  %0 = tessera.ebm.sphere_langevin_step %s, %g
      {eta = 1.000000e-02 : f64, normalized_state = false}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

func.func @clifford_bad_blade_count(%a: tensor<2x7xf32>,
                                    %b: tensor<2x7xf32>)
    -> tensor<2x7xf32> {
  // expected-error @+1 {{lhs coefficient axis must equal 2^(p+q)}}
  %0 = tessera.clifford.geometric_product %a, %b {p = 3 : i64, q = 0 : i64}
      : (tensor<2x7xf32>, tensor<2x7xf32>) -> tensor<2x7xf32>
  return %0 : tensor<2x7xf32>
}

func.func @clifford_bad_signature(%a: tensor<2x8xf32>,
                                  %b: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  // expected-error @+1 {{signature length must equal p+q}}
  %0 = tessera.clifford.outer_product %a, %b
      {p = 3 : i64, q = 0 : i64, signature = [1 : i64, -1 : i64]}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

func.func @clifford_bad_grade_mask(%a: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  // expected-error @+1 {{grade_mask entries must be in [0, p+q]}}
  %0 = tessera.clifford.grade_project %a
      {p = 3 : i64, q = 0 : i64, grade_mask = [4 : i64]}
      : (tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
