// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.7 — MLP-block 2-op fusions: matmul -> gelu and matmul -> rmsnorm.
// Both mirror the Phase 8.4.3 matmul -> softmax fusion structurally.
//
// Optimizing-Compiler Plan F2 (catalog retirement): both lower to the generic
// SYNTHESIZED epilogue kernel @tessera_apple_gpu_synth_matmul_epilogue_f32 with
// the epilogue carried as a tessera.fusion.epilogue region descriptor — one
// symbol replaces the per-epilogue hand-written matmul_gelu/matmul_rmsnorm
// kernels (uniform (A,B,O,M,N,K) signature so both declare one consistent decl).

// Runtime declaration (CHECK-DAG since order is implementation-defined).
// CHECK-DAG: func.func private @tessera_apple_gpu_synth_matmul_epilogue_f32(i64, i64, i64, i32, i32, i32)

func.func @mlp_act(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-LABEL: func.func @mlp_act
  // CHECK:       call @tessera_apple_gpu_synth_matmul_epilogue_f32
  // CHECK-SAME:  tessera.fusion.epilogue = "gelu"
  // CHECK-NOT:   tessera.matmul
  // CHECK-NOT:   tessera.gelu
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %o = "tessera.gelu"(%m) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// Use the registered rmsnorm_safe variant for the lit pass-level test.
// Plain "tessera.rmsnorm" isn't a registered Tessera dialect op (it's used
// at the Graph IR text layer in tessera.runtime); the pass handles both
// variants via separate concrete patterns. Python end-to-end tests cover
// the unregistered "tessera.rmsnorm" case through the Graph IR pipeline.

func.func @mlp_norm_safe(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-LABEL: func.func @mlp_norm_safe
  // CHECK:       call @tessera_apple_gpu_synth_matmul_epilogue_f32
  // CHECK-SAME:  tessera.fusion.epilogue = "rmsnorm"
  // CHECK-NOT:   tessera.matmul
  // CHECK-NOT:   tessera.rmsnorm
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %o = "tessera.rmsnorm_safe"(%m) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}
