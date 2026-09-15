// REQUIRES: tessera-apple-backend
// RUN: tessera-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect
//
// APPLE-MATMUL2D-1 negative fixture: the Metal 4 tensor_view / matmul2d
// contracts must REJECT (a dialect that only accepts proves nothing).

// A packed 64-wide FP8 operand is not a legal MTLTensor view: Apple requires
// a 128-byte row stride for 8-bit data (measured, macOS 27.0).
func.func @packed_fp8_stride(%a: tensor<64x64xf8E4M3FN>) -> !tessera_apple.tensor_view<f8E4M3FN> {
  // expected-error @+1 {{APPLE_TENSOR_VIEW_LAYOUT: 8/4-bit MTLTensor views need a row stride that is a multiple of 128 bytes}}
  %v = tessera_apple.gpu.tensor_view %a {byte_offset = 0 : i64, extents = array<i64: 64, 64>, strides = array<i64: 1, 64>} : tensor<64x64xf8E4M3FN> -> !tessera_apple.tensor_view<f8E4M3FN>
  return %v : !tessera_apple.tensor_view<f8E4M3FN>
}

// -----

// FP4 origins are reachable only at 256-byte multiples (Apple's 2x factor).
func.func @fp4_origin(%a: tensor<64x512xf4E2M1FN>) -> !tessera_apple.tensor_view<f4E2M1FN> {
  // expected-error @+1 {{4-bit views can only start at 256-byte origins}}
  %v = tessera_apple.gpu.tensor_view %a {byte_offset = 128 : i64, extents = array<i64: 256, 64>, strides = array<i64: 1, 512>} : tensor<64x512xf4E2M1FN> -> !tessera_apple.tensor_view<f4E2M1FN>
  return %v : !tessera_apple.tensor_view<f4E2M1FN>
}

// -----

// The view may not reinterpret its buffer's element type.
func.func @view_storage_mismatch(%a: tensor<64x128xf16>) -> !tessera_apple.tensor_view<bf16> {
  // expected-error @+1 {{APPLE_TENSOR_VIEW_STORAGE}}
  %v = tessera_apple.gpu.tensor_view %a {byte_offset = 0 : i64, extents = array<i64: 128, 64>, strides = array<i64: 1, 128>} : tensor<64x128xf16> -> !tessera_apple.tensor_view<bf16>
  return %v : !tessera_apple.tensor_view<bf16>
}

// -----

// bf16 x f16 is not an MPP operand pair and is refused, not converted.
func.func @mixed_pair(%a: tensor<64x128xbf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf32> {
  %va = tessera_apple.gpu.tensor_view %a {byte_offset = 0 : i64, extents = array<i64: 128, 64>, strides = array<i64: 1, 128>} : tensor<64x128xbf16> -> !tessera_apple.tensor_view<bf16>
  %vb = tessera_apple.gpu.tensor_view %b {byte_offset = 0 : i64, extents = array<i64: 64, 128>, strides = array<i64: 1, 64>} : tensor<128x64xf16> -> !tessera_apple.tensor_view<f16>
  // expected-error @+1 {{APPLE_MATMUL2D_PAIR_UNSUPPORTED}}
  %r = tessera_apple.gpu.matmul2d %va, %vb {tile_m = 64 : i64, tile_n = 64 : i64, simdgroups = 4 : i64, accumulate = "f32"} : !tessera_apple.tensor_view<bf16>, !tessera_apple.tensor_view<f16> -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// -----

// An f16 result is a re-rounded accumulator, not the fp32 contract.
func.func @f16_accum(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %va = tessera_apple.gpu.tensor_view %a {byte_offset = 0 : i64, extents = array<i64: 128, 64>, strides = array<i64: 1, 128>} : tensor<64x128xf16> -> !tessera_apple.tensor_view<f16>
  %vb = tessera_apple.gpu.tensor_view %b {byte_offset = 0 : i64, extents = array<i64: 64, 128>, strides = array<i64: 1, 64>} : tensor<128x64xf16> -> !tessera_apple.tensor_view<f16>
  // expected-error @+1 {{APPLE_MATMUL2D_ACCUM}}
  %r = tessera_apple.gpu.matmul2d %va, %vb {tile_m = 64 : i64, tile_n = 64 : i64, simdgroups = 4 : i64, accumulate = "f32"} : !tessera_apple.tensor_view<f16>, !tessera_apple.tensor_view<f16> -> tensor<64x64xf16>
  return %r : tensor<64x64xf16>
}

// -----

// K disagreement between the two views.
func.func @k_mismatch(%a: tensor<64x128xf16>, %b: tensor<96x64xf16>) -> tensor<64x64xf32> {
  %va = tessera_apple.gpu.tensor_view %a {byte_offset = 0 : i64, extents = array<i64: 128, 64>, strides = array<i64: 1, 128>} : tensor<64x128xf16> -> !tessera_apple.tensor_view<f16>
  %vb = tessera_apple.gpu.tensor_view %b {byte_offset = 0 : i64, extents = array<i64: 64, 96>, strides = array<i64: 1, 64>} : tensor<96x64xf16> -> !tessera_apple.tensor_view<f16>
  // expected-error @+1 {{APPLE_MATMUL2D_SHAPE: a's inner extent (K=128) must equal b's outer extent (96)}}
  %r = tessera_apple.gpu.matmul2d %va, %vb {tile_m = 64 : i64, tile_n = 64 : i64, simdgroups = 4 : i64, accumulate = "f32"} : !tessera_apple.tensor_view<f16>, !tessera_apple.tensor_view<f16> -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}
