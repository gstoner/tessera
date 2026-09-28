// RUN: tessera-opt %s -split-input-file -verify-diagnostics
//
// stencil.define well-formedness is enforced by the op verifier (the single
// authority, src/compiler/ir/TesseraOps.cpp), so it fails at parse -- before
// and independent of -tessera-stencil-lower, which used to be the only place
// it was checked (SMALL-CORRECTNESS-GAPS-2026-09-27).

func.func @missing_coeff(%arg0: tensor<?x?xf32>) {
  // expected-error @+1 {{requires explicit non-empty 'coeffs' array}}
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>]
  } : () -> index
  return
}

// -----

func.func @mismatched_coeffs(%arg0: tensor<?x?xf32>) {
  // expected-error @+1 {{requires one coefficient per tap; got 2 taps and 1 coefficients}}
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>,
            dense<[1, 0]> : tensor<2xi64>],
    coeffs = [1.0 : f64]
  } : () -> index
  return
}

// -----

func.func @wrong_coeff_dtype(%arg0: tensor<?x?xf32>) {
  // expected-error @+1 {{coefficient 0 must be a finite canonical f64 value}}
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>],
    coeffs = [1.0 : f32]
  } : () -> index
  return
}

// -----

func.func @ragged_tap_rank(%arg0: tensor<?x?xf32>) {
  // expected-error @+1 {{tap 1 has rank 1, expected 2}}
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>, dense<[1]> : tensor<1xi64>],
    coeffs = [1.0 : f64, 1.0 : f64]
  } : () -> index
  return
}

// -----

func.func @missing_taps() {
  // expected-error @+1 {{requires a non-empty 'taps' array}}
  %st = "tessera.neighbors.stencil.define"() {
    coeffs = [1.0 : f64]
  } : () -> index
  return
}

// -----

func.func @non_integer_tap() {
  // expected-error @+1 {{tap 0 must be a non-empty rank-1 dense integer vector}}
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0.0, 1.0]> : tensor<2xf32>],
    coeffs = [1.0 : f64]
  } : () -> index
  return
}

// -----

func.func @neighbor_read_without_delta(%halo: !tessera.neighbors.halo) -> f32 {
  // expected-error @+1 {{requires a 'delta' attribute}}
  %v = "tessera.neighbors.neighbor.read"(%halo) : (!tessera.neighbors.halo) -> f32
  return %v : f32
}

// -----

func.func @neighbor_read_empty_delta(%halo: !tessera.neighbors.halo) -> f32 {
  // expected-error @+1 {{requires 'delta' array to be non-empty and all integers}}
  %v = "tessera.neighbors.neighbor.read"(%halo) {delta = []}
      : (!tessera.neighbors.halo) -> f32
  return %v : f32
}

// -----

func.func @neighbor_read_float_delta(%halo: !tessera.neighbors.halo) -> f32 {
  // expected-error @+1 {{requires 'delta' to be a dense integer vector or an array of integers}}
  %v = "tessera.neighbors.neighbor.read"(%halo) {delta = 1.0 : f32}
      : (!tessera.neighbors.halo) -> f32
  return %v : f32
}

// -----

// Positive controls: both encodings tessera-halo-infer reads are accepted.
func.func @neighbor_read_valid(%halo: !tessera.neighbors.halo) -> (f32, f32) {
  %a = "tessera.neighbors.neighbor.read"(%halo) {delta = dense<[1, 0]> : tensor<2xi64>}
      : (!tessera.neighbors.halo) -> f32
  %b = "tessera.neighbors.neighbor.read"(%halo) {delta = [0, -1]}
      : (!tessera.neighbors.halo) -> f32
  %st = "tessera.neighbors.stencil.define"() {
    taps = [dense<[0, 0]> : tensor<2xi64>, dense<[1, 0]> : tensor<2xi64>],
    coeffs = [-2.0 : f64, 1.0 : f64]
  } : () -> index
  return %a, %b : f32, f32
}
