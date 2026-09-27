// RUN: tessera-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// ROCM-FP8-BLOCKSCALE-1: `tile.fragment_scaled_accumulate` joins one scale
// group's isolated fp32 partial into the running accumulator. Its operands are
// typed: an f32 accumulator fragment and rank-1 f32 scale buffers.

// CHECK-LABEL: func.func @join
// CHECK: tile.fragment_scaled_accumulate %{{.*}}, %{{.*}} scales(%{{.*}}, %{{.*}}) at(%{{.*}}, %{{.*}}) group(%{{.*}}, %{{.*}}) bounds(%{{.*}}, %{{.*}}) {scale_n = 128 : i64}
func.func @join(%sa: memref<?xf32>, %sb: memref<?xf32>, %i: index) {
  %acc = tile.fragment_zero : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
  %part = tile.fragment_zero : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
  %0 = tile.fragment_scaled_accumulate %acc, %part scales(%sa, %sb) at(%i, %i) group(%i, %i) bounds(%i, %i) {scale_n = 128 : i64} : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">, memref<?xf32>, memref<?xf32>
  return
}

// -----

func.func @int_scales(%sa: memref<?xi32>, %sb: memref<?xf32>, %i: index) {
  %acc = tile.fragment_zero : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
  // expected-error @+1 {{TILE_FRAGMENT_SCALED_ACCUMULATE_SCALE}}
  %0 = tile.fragment_scaled_accumulate %acc, %acc scales(%sa, %sb) at(%i, %i) group(%i, %i) bounds(%i, %i) {scale_n = 1 : i64} : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">, memref<?xi32>, memref<?xf32>
  return
}

// -----

func.func @zero_block(%sa: memref<?xf32>, %sb: memref<?xf32>, %i: index) {
  %acc = tile.fragment_zero : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
  // expected-error @+1 {{TILE_FRAGMENT_SCALED_ACCUMULATE_SCALE}}
  %0 = tile.fragment_scaled_accumulate %acc, %acc scales(%sa, %sb) at(%i, %i) group(%i, %i) bounds(%i, %i) {scale_n = 0 : i64} : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">, memref<?xf32>, memref<?xf32>
  return
}

// -----

func.func @int_accumulator(%sa: memref<?xf32>, %sb: memref<?xf32>, %i: index) {
  %acc = tile.fragment_zero : !tile.fragment<m = 16, n = 16, k = 16, elem = "i32", acc = "i32", role = "acc", layout = "row_major", family = "wmma">
  // expected-error @+1 {{TILE_FRAGMENT_SCALED_ACCUMULATE_TYPE}}
  %0 = tile.fragment_scaled_accumulate %acc, %acc scales(%sa, %sb) at(%i, %i) group(%i, %i) bounds(%i, %i) {scale_n = 1 : i64} : !tile.fragment<m = 16, n = 16, k = 16, elem = "i32", acc = "i32", role = "acc", layout = "row_major", family = "wmma">, memref<?xf32>, memref<?xf32>
  return
}
