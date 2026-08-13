// RUN: not tessera-opt %s -tessera-stencil-lower 2>&1 | FileCheck %s

func.func @missing_coeff(%arg0: tensor<?x?xf32>) {
  // CHECK: error: stencil.define requires explicit non-empty 'coeffs' array
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>]
  } : () -> index
  return
}

func.func @mismatched_coeffs(%arg0: tensor<?x?xf32>) {
  // CHECK: error: stencil.define requires one coefficient per tap
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>,
            dense<[1, 0]> : tensor<2xi64>],
    coeffs = [1.0 : f64]
  } : () -> index
  return
}

func.func @wrong_coeff_dtype(%arg0: tensor<?x?xf32>) {
  // CHECK: error: stencil.define coefficient 0 must be a finite canonical f64 value
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>],
    coeffs = [1.0 : f32]
  } : () -> index
  return
}

func.func @ragged_tap_rank(%arg0: tensor<?x?xf32>) {
  // CHECK: error: stencil.define tap 1 has rank 1, expected 2
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>, dense<[1]> : tensor<1xi64>],
    coeffs = [1.0 : f64, 1.0 : f64]
  } : () -> index
  return
}
