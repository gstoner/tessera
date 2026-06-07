// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V9 (2026-06-07) — control-flow payload/bounds verifiers, closed
// trivial-stub resource/scalar contracts (Arch*/KVCacheCreate/RingCreate), and
// MoR / quantize-dequantize / FFT-family checks.

// ─── control_for — positive ─────────────────────────────────────────────────
func.func private @body()
// CHECK-LABEL: func.func @control_for_ok
func.func @control_for_ok(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init) {body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
func.func private @body()
func.func @control_for_step0(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  // expected-error @+1 {{step must be non-zero}}
  %r = "tessera.control_for"(%init) {body = @body, start = 0 : i64, stop = 8 : i64, step = 0 : i64} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
func.func private @body()
func.func @control_for_carry_mismatch(%init: tensor<1x8xf32>) -> tensor<2x8xf32> {
  // expected-error @+1 {{loop-carried type mismatch}}
  %r = "tessera.control_for"(%init) {body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64} : (tensor<1x8xf32>) -> tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----
// ─── control_while — max_iters must be positive ─────────────────────────────
func.func private @wb()
func.func private @wc()
func.func @control_while_maxiters(%init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // expected-error @+1 {{max_iters must be positive}}
  %r = "tessera.control_while"(%init) {body = @wb, cond = @wc, carry_arg_index = 0 : i64, max_iters = 0 : i64} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
// ─── control_if — flag index out of range ───────────────────────────────────
func.func private @tb()
func.func private @eb()
func.func @control_if_flag(%flag: tensor<1xf32>) -> tensor<1x8xf32> {
  // expected-error @+1 {{flag_arg_index out of range}}
  %r = "tessera.control_if"(%flag) {then_branch = @tb, else_branch = @eb, flag_arg_index = 3 : i64} : (tensor<1xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── KVCacheCreate — positive / negative ────────────────────────────────────
// CHECK-LABEL: func.func @kv_create_ok
func.func @kv_create_ok() -> !tessera.kv_cache {
  %c = "tessera.kv_cache.create"() {max_seq = 2048 : i64, head_dim = 128 : i64} : () -> !tessera.kv_cache
  return %c : !tessera.kv_cache
}

// -----
func.func @kv_create_bad() -> !tessera.kv_cache {
  // expected-error @+1 {{max_seq must be positive}}
  %c = "tessera.kv_cache.create"() {max_seq = 0 : i64, head_dim = 128 : i64} : () -> !tessera.kv_cache
  return %c : !tessera.kv_cache
}

// -----
// ─── RingCreate — capacity must be positive ─────────────────────────────────
func.func @ring_bad() -> !tessera.ring {
  // expected-error @+1 {{capacity must be positive}}
  %r = "tessera.ring.create"() {capacity = -1 : i64} : () -> !tessera.ring
  return %r : !tessera.ring
}

// -----
// ─── MorRouter — max_depth must be positive ─────────────────────────────────
func.func @mor_router_bad(%x: tensor<4x8xf32>, %w: tensor<8x1xf32>) -> tensor<4x1xf32> {
  // expected-error @+1 {{max_depth must be positive}}
  %d = "tessera.mor_router"(%x, %w) {max_depth = 0 : i64} : (tensor<4x8xf32>, tensor<8x1xf32>) -> tensor<4x1xf32>
  return %d : tensor<4x1xf32>
}

// -----
// ─── QuantizeFP8 — positive / bad format / shape mismatch ───────────────────
// CHECK-LABEL: func.func @quant_fp8_ok
func.func @quant_fp8_ok(%x: tensor<4x8xf32>) -> (tensor<4x8xf8E4M3FN>, f32) {
  %q, %s = "tessera.quantize_fp8"(%x) {format = "e4m3"} : (tensor<4x8xf32>) -> (tensor<4x8xf8E4M3FN>, f32)
  return %q, %s : tensor<4x8xf8E4M3FN>, f32
}

// -----
func.func @quant_fp8_badfmt(%x: tensor<4x8xf32>) -> (tensor<4x8xf8E4M3FN>, f32) {
  // expected-error @+1 {{format has unsupported value 'int8'}}
  %q, %s = "tessera.quantize_fp8"(%x) {format = "int8"} : (tensor<4x8xf32>) -> (tensor<4x8xf8E4M3FN>, f32)
  return %q, %s : tensor<4x8xf8E4M3FN>, f32
}

// -----
func.func @quant_fp8_shape(%x: tensor<4x8xf32>) -> (tensor<4x16xf8E4M3FN>, f32) {
  // expected-error @+1 {{quantize_fp8 shapes must match}}
  %q, %s = "tessera.quantize_fp8"(%x) {format = "e4m3"} : (tensor<4x8xf32>) -> (tensor<4x16xf8E4M3FN>, f32)
  return %q, %s : tensor<4x16xf8E4M3FN>, f32
}

// -----
// ─── FFT — axis out of range / positive ─────────────────────────────────────
// CHECK-LABEL: func.func @fft_ok
func.func @fft_ok(%x: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %y = "tessera.fft"(%x) {axis = -1 : i64} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %y : tensor<4x16xf32>
}

// -----
func.func @fft_axis(%x: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // expected-error @+1 {{fft axis out of range}}
  %y = "tessera.fft"(%x) {axis = 5 : i64} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %y : tensor<4x16xf32>
}
