// RUN: tessera-opt %s -split-input-file -verify-diagnostics
//
// AR-DECODE-PROGRAM-1 S1a: the registered verifier refuses every shape and
// dtype outside the in-place paged write contract.

func.func @result_must_alias_pages(%p: tensor<4x16x2x16xf32>, %t: tensor<4xi32>, %v: tensor<1x2x16xf32>, %pos: tensor<1xi32>) {
  // expected-error @+1 {{result must be the updated pages type}}
  %u = tessera.paged_kv_write %p, %t, %v, %pos : (tensor<4x16x2x16xf32>, tensor<4xi32>, tensor<1x2x16xf32>, tensor<1xi32>) -> tensor<4x16x2x8xf32>
  return
}

// -----

func.func @heads_must_match(%p: tensor<4x16x2x16xf32>, %t: tensor<4xi32>, %v: tensor<1x3x16xf32>, %pos: tensor<1xi32>) {
  // expected-error @+1 {{value heads/head_dim disagree with the pages}}
  %u = tessera.paged_kv_write %p, %t, %v, %pos : (tensor<4x16x2x16xf32>, tensor<4xi32>, tensor<1x3x16xf32>, tensor<1xi32>) -> tensor<4x16x2x16xf32>
  return
}

// -----

func.func @tokens_within_capacity(%p: tensor<4x16x2x16xf32>, %t: tensor<2xi32>, %v: tensor<33x2x16xf32>, %pos: tensor<1xi32>) {
  // expected-error @+1 {{token count exceeds the logical page capacity}}
  %u = tessera.paged_kv_write %p, %t, %v, %pos : (tensor<4x16x2x16xf32>, tensor<2xi32>, tensor<33x2x16xf32>, tensor<1xi32>) -> tensor<4x16x2x16xf32>
  return
}

// -----

func.func @position_is_one_i32(%p: tensor<4x16x2x16xf32>, %t: tensor<4xi32>, %v: tensor<1x2x16xf32>, %pos: tensor<1xi64>) {
  // expected-error @+1 {{requires static f32 pages[P,PS,H,D], i32 table[LP], f32 values[T,H,D] and i32 position[1]}}
  %u = tessera.paged_kv_write %p, %t, %v, %pos : (tensor<4x16x2x16xf32>, tensor<4xi32>, tensor<1x2x16xf32>, tensor<1xi64>) -> tensor<4x16x2x16xf32>
  return
}

// -----

func.func @pages_are_f32(%p: tensor<4x16x2x16xf16>, %t: tensor<4xi32>, %v: tensor<1x2x16xf32>, %pos: tensor<1xi32>) {
  // expected-error @+1 {{requires static f32 pages[P,PS,H,D]}}
  %u = tessera.paged_kv_write %p, %t, %v, %pos : (tensor<4x16x2x16xf16>, tensor<4xi32>, tensor<1x2x16xf32>, tensor<1xi32>) -> tensor<4x16x2x16xf16>
  return
}
