// RUN: tessera-opt %s --tessera-lower-to-apple_gpu-runtime --allow-unregistered-dialect | FileCheck %s

// 2026-05-29 — Apple GPU MPSGraph Tier-1 lane lowering.
//
// Verifies that the Tier-1 activation / normalization ops lower to func.call
// into the MetalPerformanceShadersGraph-backed runtime symbols:
//   * unary activations  -> tessera_apple_gpu_mpsgraph_unary_f32 (op-coded)
//   * silu_mul           -> tessera_apple_gpu_mpsgraph_binary_f32 (opcode 6)
//   * layer_norm/rmsnorm -> tessera_apple_gpu_{layer_norm,rmsnorm_gpu}_f32
//                           with null (0) gamma/beta (the ops are unweighted)
//   * log_softmax        -> tessera_apple_gpu_log_softmax_f32
// The op-coded unary symbol carries the opcode matching mpsg_unary_node in
// apple_gpu_runtime.mm (silu = 4, relu = 0).

// ----------------------------------------------------------------------------
// silu -> unary opcode 4
// ----------------------------------------------------------------------------
func.func @silu(%x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @silu
  // CHECK:       %[[OP:.*]] = arith.constant 4 : i32
  // CHECK:       call @tessera_apple_gpu_mpsgraph_unary_f32(%[[OP]],
  // CHECK-NOT:   tessera.silu
  %0 = "tessera.silu"(%x) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// relu -> unary opcode 0
// ----------------------------------------------------------------------------
func.func @relu(%x: tensor<4x32xf32>) -> tensor<4x32xf32> {
  // CHECK-LABEL: func.func @relu
  // CHECK:       %[[OP:.*]] = arith.constant 0 : i32
  // CHECK:       call @tessera_apple_gpu_mpsgraph_unary_f32(%[[OP]],
  %0 = "tessera.relu"(%x) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// ----------------------------------------------------------------------------
// silu_mul -> binary opcode 6 (silu(a) * b)
// ----------------------------------------------------------------------------
func.func @silu_mul(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @silu_mul
  // CHECK:       %[[OP:.*]] = arith.constant 6 : i32
  // CHECK:       call @tessera_apple_gpu_mpsgraph_binary_f32(%[[OP]],
  // CHECK-NOT:   tessera.silu_mul
  %0 = "tessera.silu_mul"(%a, %b) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// layer_norm -> layer_norm_f32 with null (0) gamma/beta (unweighted).
// ----------------------------------------------------------------------------
func.func @layer_norm(%x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @layer_norm
  // CHECK:       call @tessera_apple_gpu_layer_norm_f32
  // CHECK-NOT:   tessera.layer_norm
  %0 = "tessera.layer_norm"(%x) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// rmsnorm -> rmsnorm_gpu_f32 (unweighted)
// ----------------------------------------------------------------------------
func.func @rmsnorm(%x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @rmsnorm
  // CHECK:       call @tessera_apple_gpu_rmsnorm_gpu_f32
  %0 = "tessera.rmsnorm"(%x) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// ----------------------------------------------------------------------------
// log_softmax -> log_softmax_f32
// ----------------------------------------------------------------------------
func.func @log_softmax(%x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @log_softmax
  // CHECK:       call @tessera_apple_gpu_log_softmax_f32
  %0 = "tessera.log_softmax"(%x) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
