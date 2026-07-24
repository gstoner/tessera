// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_cpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.2 — Apple CPU native execution path. Verifies that the runtime
// pipeline lowers a static-shape f32 matmul to a func.call into the Accelerate
// runtime shim. This is the executable counterpart to the artifact-only
// apple_cpu_lowering.mlir fixture.

// CHECK-LABEL: func.func private @tessera_apple_cpu_gemm_f32
// CHECK-SAME:  (i64, i64, i64, i32, i32, i32)

func.func @gemm_f32(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>) -> tensor<128x64xf32> {
  // CHECK-LABEL: func.func @gemm_f32
  // CHECK:       bufferization.to_buffer
  // CHECK:       memref.extract_aligned_pointer_as_index
  // CHECK:       arith.index_cast
  // CHECK:       memref.alloc()
  // CHECK:       call @tessera_apple_cpu_gemm_f32
  // CHECK-NOT:   tessera.matmul
  %C = "tessera.matmul"(%A, %B) : (tensor<128x256xf32>, tensor<256x64xf32>) -> tensor<128x64xf32>
  return %C : tensor<128x64xf32>
}

// Negative case: dynamic shapes are rejected by the Phase 8.2 path; the op
// stays unchanged, the runtime decl is not emitted.

// CHECK-LABEL: func.func @gemm_dynamic
// CHECK:       tessera.matmul
// CHECK-NOT:   call @tessera_apple_cpu_gemm_f32

func.func @gemm_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %C = "tessera.matmul"(%A, %B) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %C : tensor<?x?xf32>
}
