// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Decision #19 — the Apple Target IR fusion pass emits a first-class fusion
// descriptor on the fused call (`tessera.fusion.kernel` + `tessera.fusion.source`)
// and *consumes* the compiler's `tessera.fusion.intent` when present:
//   * intent stamped (the canonical compile recognized the chain) → source="descriptor"
//   * no intent (legacy IR) → the structural walk re-discovers it → source="rediscovered"
// Both fuse to the same matmul_softmax_matmul kernel.

// CHECK-LABEL: func.func @descriptor_driven
func.func @descriptor_driven(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>,
                             %C: tensor<32x8xf32>) -> tensor<8x8xf32> {
  // CHECK: call @tessera_apple_gpu_matmul_softmax_matmul_f32
  // CHECK-SAME: tessera.fusion.kernel = "matmul_softmax_matmul"
  // CHECK-SAME: tessera.fusion.source = "descriptor"
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %o  = "tessera.matmul"(%p, %C) {tessera.fusion.intent = "matmul_softmax_matmul"}
        : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
  return %o : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @rediscovered
func.func @rediscovered(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>,
                        %C: tensor<32x8xf32>) -> tensor<8x8xf32> {
  // CHECK: call @tessera_apple_gpu_matmul_softmax_matmul_f32
  // CHECK-SAME: tessera.fusion.source = "rediscovered"
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %o  = "tessera.matmul"(%p, %C) : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
  return %o : tensor<8x8xf32>
}

// matmul→gelu: descriptor-driven.
// CHECK-LABEL: func.func @gelu_descriptor
func.func @gelu_descriptor(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK: call @tessera_apple_gpu_matmul_gelu_f32
  // CHECK-SAME: tessera.fusion.kernel = "matmul_gelu"
  // CHECK-SAME: tessera.fusion.source = "descriptor"
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %g = "tessera.gelu"(%m) {tessera.fusion.intent = "matmul_gelu"} : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %g : tensor<8x32xf32>
}

// matmul→rmsnorm: re-discovered (no intent).
// CHECK-LABEL: func.func @rmsnorm_rediscovered
func.func @rmsnorm_rediscovered(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK: call @tessera_apple_gpu_matmul_rmsnorm_f32
  // CHECK-SAME: tessera.fusion.kernel = "matmul_rmsnorm"
  // CHECK-SAME: tessera.fusion.source = "rediscovered"
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %r = "tessera.rmsnorm"(%m) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %r : tensor<8x32xf32>
}
