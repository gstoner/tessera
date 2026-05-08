// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.3 — Apple GPU first multi-op MSL fusion (matmul -> softmax).
// Verifies that the runtime pipeline collapses a 2-op SSA chain into a
// single func.call into the fused runtime symbol. The matmul op is gone
// from the rewritten module (no per-op matmul or softmax calls remain).

// CHECK-LABEL: func.func private @tessera_apple_gpu_matmul_softmax_f32
// CHECK-SAME:  (i64, i64, i64, i32, i32, i32)

func.func @fused_static(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-LABEL: func.func @fused_static
  // CHECK:       call @tessera_apple_gpu_matmul_softmax_f32
  // CHECK-NOT:   tessera.matmul
  // CHECK-NOT:   tessera.softmax
  // CHECK-NOT:   call @tessera_apple_gpu_mps_matmul_f32
  // CHECK-NOT:   call @tessera_apple_gpu_softmax_f32
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %o = "tessera.softmax"(%m) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// Negative case: the matmul result has multiple uses — the chain must
// not fire because folding the intermediate would change observable
// semantics. Both per-op runtime calls should appear instead.

// CHECK-LABEL: func.func @matmul_with_extra_use
// CHECK-DAG:   call @tessera_apple_gpu_mps_matmul_f32
// CHECK-DAG:   call @tessera_apple_gpu_softmax_f32
// CHECK-NOT:   call @tessera_apple_gpu_matmul_softmax_f32

func.func @matmul_with_extra_use(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> (tensor<8x32xf32>, tensor<8x32xf32>) {
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %o = "tessera.softmax"(%m) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %m, %o : tensor<8x32xf32>, tensor<8x32xf32>
}

// Negative case: N > 256 falls out of the GPU kernel envelope.

// CHECK-LABEL: func.func @fused_n_too_big
// CHECK-NOT:   call @tessera_apple_gpu_matmul_softmax_f32

func.func @fused_n_too_big(%A: tensor<2x4xf32>, %B: tensor<4x512xf32>) -> tensor<2x512xf32> {
  %m = "tessera.matmul"(%A, %B) : (tensor<2x4xf32>, tensor<4x512xf32>) -> tensor<2x512xf32>
  %o = "tessera.softmax"(%m) : (tensor<2x512xf32>) -> tensor<2x512xf32>
  return %o : tensor<2x512xf32>
}
