// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.1 — Apple GPU custom MSL flash-attention forward path. Verifies
// that the runtime pipeline lowers a static-shape rank-3 f32 tessera.flash_attn
// op to a func.call into the apple_gpu_runtime.mm shim's
// tessera_apple_gpu_flash_attn_f32 symbol. tessera.flash_attn IS a registered
// dialect op (Tessera_FlashAttnOp), so a pass-level lit test can use it
// directly — same shape as the matmul fixture, with one additional f32 scale
// argument and an i32 causal flag in the call site.

// CHECK-LABEL: func.func private @tessera_apple_gpu_flash_attn_f32
// CHECK-SAME:  (i64, i64, i64, i64, i32, i32, i32, i32, f32, i32)

func.func @flash_static(%Q: tensor<2x8x16xf32>, %K: tensor<2x8x16xf32>, %V: tensor<2x8x16xf32>) -> tensor<2x8x16xf32> {
  // CHECK-LABEL: func.func @flash_static
  // CHECK:       bufferization.to_buffer
  // CHECK:       memref.extract_aligned_pointer_as_index
  // CHECK:       arith.index_cast
  // CHECK:       memref.alloc()
  // CHECK:       call @tessera_apple_gpu_flash_attn_f32
  // CHECK-NOT:   tessera.flash_attn
  %O = "tessera.flash_attn"(%Q, %K, %V) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {causal = false, head_dim = 16 : i64} : (tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  return %O : tensor<2x8x16xf32>
}

// Negative case: dynamic shapes are rejected by the Phase 8.4.1 path.

// CHECK-LABEL: func.func @flash_dynamic
// CHECK:       tessera.flash_attn
// CHECK-NOT:   call @tessera_apple_gpu_flash_attn_f32

func.func @flash_dynamic(%Q: tensor<?x?x?xf32>, %K: tensor<?x?x?xf32>, %V: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %O = "tessera.flash_attn"(%Q, %K, %V) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {causal = false, head_dim = 16 : i64} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %O : tensor<?x?x?xf32>
}

// Negative case: head_dim > 256 is out of envelope and falls back.

// CHECK-LABEL: func.func @flash_head_too_big
// CHECK:       tessera.flash_attn
// CHECK-NOT:   call @tessera_apple_gpu_flash_attn_f32

func.func @flash_head_too_big(%Q: tensor<1x4x512xf32>, %K: tensor<1x4x512xf32>, %V: tensor<1x4x512xf32>) -> tensor<1x4x512xf32> {
  %O = "tessera.flash_attn"(%Q, %K, %V) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {causal = false, head_dim = 512 : i64} : (tensor<1x4x512xf32>, tensor<1x4x512xf32>, tensor<1x4x512xf32>) -> tensor<1x4x512xf32>
  return %O : tensor<1x4x512xf32>
}
