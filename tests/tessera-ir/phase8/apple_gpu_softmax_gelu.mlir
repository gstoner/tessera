// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.2 — Apple GPU custom MSL softmax + gelu kernels. Verifies that
// the runtime pipeline lowers static-shape rank-2 f32 tessera.softmax and
// tessera.gelu ops to func.calls into their respective runtime shim
// symbols. Both ops are in the registered Tessera dialect.

// Both runtime decls must appear in the module preamble. Ordering between
// the two is implementation-defined, so use CHECK-DAG.
// CHECK-DAG: func.func private @tessera_apple_gpu_softmax_f32_status(i64, i64, i32, i32) -> i32
// CHECK-DAG: func.func private @tessera_apple_gpu_gelu_f32_status(i64, i64, i32) -> i32

func.func @softmax_static(%X: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @softmax_static
  // CHECK:       call @tessera_apple_gpu_softmax_f32_status
  // CHECK:       cf.assert
  // CHECK-NOT:   tessera.softmax
  %Out = "tessera.softmax"(%X) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %Out : tensor<8x16xf32>
}

func.func @gelu_static(%X: tensor<4x32xf32>) -> tensor<4x32xf32> {
  // CHECK-LABEL: func.func @gelu_static
  // CHECK:       call @tessera_apple_gpu_gelu_f32_status
  // CHECK:       cf.assert
  // CHECK-NOT:   tessera.gelu
  %Out = "tessera.gelu"(%X) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  return %Out : tensor<4x32xf32>
}

// Negative case: dynamic shapes are rejected.

// CHECK-LABEL: func.func @softmax_dynamic
// CHECK:       tessera.softmax
// CHECK-NOT:   call @tessera_apple_gpu_softmax_f32_status

func.func @softmax_dynamic(%X: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %Out = "tessera.softmax"(%X) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %Out : tensor<?x?xf32>
}

// Dynamic GELU carries dimensions and checks its native ABI before allocation.

// CHECK-LABEL: func.func @gelu_dynamic
// CHECK:       tensor.dim
// CHECK:       cf.assert
// CHECK:       memref.alloc
// CHECK:       call @tessera_apple_gpu_gelu_f32_status
// CHECK:       cf.assert

func.func @gelu_dynamic(%X: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %Out = "tessera.gelu"(%X) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %Out : tensor<?x?xf32>
}
