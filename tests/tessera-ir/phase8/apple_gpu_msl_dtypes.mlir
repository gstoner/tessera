// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.4.1 — Apple GPU custom MSL kernels with fp16 / bf16 dtype
// variants for softmax and gelu. Verifies that the runtime pipeline picks
// the right runtime symbol per (kernel, dtype) pair. All three dtypes
// share the same i64 + i32 ABI shape per kernel — element type is encoded
// in the symbol name only.
//
// rope dtype dispatch is exercised by Python unit tests instead — the
// tessera.rope op is not registered in the Tessera dialect, which prevents
// pass-level lit testing of it.

// Runtime declarations: one per (kernel, dtype) pair. Order between
// declarations is implementation-defined, so use CHECK-DAG.
// CHECK-DAG: func.func private @tessera_apple_gpu_softmax_f32(i64, i64, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_softmax_f16(i64, i64, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_softmax_bf16(i64, i64, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_gelu_f32(i64, i64, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_gelu_f16(i64, i64, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_gelu_bf16(i64, i64, i32)

func.func @softmax_f16(%X: tensor<8x16xf16>) -> tensor<8x16xf16> {
  // CHECK-LABEL: func.func @softmax_f16
  // CHECK:       call @tessera_apple_gpu_softmax_f16
  %Out = "tessera.softmax"(%X) : (tensor<8x16xf16>) -> tensor<8x16xf16>
  return %Out : tensor<8x16xf16>
}

func.func @softmax_bf16(%X: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
  // CHECK-LABEL: func.func @softmax_bf16
  // CHECK:       call @tessera_apple_gpu_softmax_bf16
  %Out = "tessera.softmax"(%X) : (tensor<8x16xbf16>) -> tensor<8x16xbf16>
  return %Out : tensor<8x16xbf16>
}

func.func @gelu_f16(%X: tensor<8x16xf16>) -> tensor<8x16xf16> {
  // CHECK-LABEL: func.func @gelu_f16
  // CHECK:       call @tessera_apple_gpu_gelu_f16
  %Out = "tessera.gelu"(%X) : (tensor<8x16xf16>) -> tensor<8x16xf16>
  return %Out : tensor<8x16xf16>
}

func.func @gelu_bf16(%X: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
  // CHECK-LABEL: func.func @gelu_bf16
  // CHECK:       call @tessera_apple_gpu_gelu_bf16
  %Out = "tessera.gelu"(%X) : (tensor<8x16xbf16>) -> tensor<8x16xbf16>
  return %Out : tensor<8x16xbf16>
}

// f32 cases — these paths shouldn't regress.

func.func @softmax_f32(%X: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @softmax_f32
  // CHECK:       call @tessera_apple_gpu_softmax_f32
  %Out = "tessera.softmax"(%X) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %Out : tensor<8x16xf32>
}

func.func @gelu_f32(%X: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @gelu_f32
  // CHECK:       call @tessera_apple_gpu_gelu_f32
  %Out = "tessera.gelu"(%X) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %Out : tensor<8x16xf32>
}
