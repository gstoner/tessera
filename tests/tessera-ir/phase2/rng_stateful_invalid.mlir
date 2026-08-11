// RUN: tessera-opt %s -verify-diagnostics

// Negative coverage for the implicit-stream stateful RNG ops
// (tessera.rng_uniform / tessera.rng_normal). The verifier must reject
// shape-attribute drift, malformed distribution parameters, and non-float
// sample types — a dialect that only ever accepts proves nothing.

func.func @shape_mismatch() -> tensor<4xf32> {
  // expected-error@+1 {{shape attribute must match the result type shape}}
  %s = "tessera.rng_uniform"() {shape = [8], seed = 7 : i64}
      : () -> tensor<4xf32>
  return %s : tensor<4xf32>
}

func.func @negative_shape_entry() -> tensor<4xf32> {
  // expected-error@+1 {{shape must contain only non-negative integers}}
  %s = "tessera.rng_uniform"() {shape = [-4]}
      : () -> tensor<4xf32>
  return %s : tensor<4xf32>
}

func.func @inverted_bounds() -> tensor<4xf32> {
  // expected-error@+1 {{requires finite low < high}}
  %s = "tessera.rng_uniform"()
      {shape = [4], low = 2.0 : f64, high = 1.0 : f64}
      : () -> tensor<4xf32>
  return %s : tensor<4xf32>
}

func.func @bad_stddev() -> tensor<4xf32> {
  // expected-error@+1 {{requires finite mean and stddev > 0}}
  %s = "tessera.rng_normal"()
      {shape = [4], stddev = 0.0 : f64}
      : () -> tensor<4xf32>
  return %s : tensor<4xf32>
}

func.func @non_float_sample() -> tensor<4xi32> {
  // expected-error@+1 {{sample must be a ranked floating tensor}}
  %s = "tessera.rng_uniform"() {shape = [4]}
      : () -> tensor<4xi32>
  return %s : tensor<4xi32>
}
