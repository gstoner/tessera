// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// attn_bias substrate (DFlash sliding-layer / general additive mask). A rank-3
// flash_attn carrying an optional attn_bias operand (B, Sq, Sk) lowers to the
// bias-aware Apple GPU runtime symbol tessera_apple_gpu_flash_attn_bias_f32,
// which applies softmax(scale*Q*K^T + bias)*V on MPSGraph. The bias-free path
// is unchanged and must still hit the plain flash_attn_f32 symbol.

// CHECK-LABEL: func.func private @tessera_apple_gpu_flash_attn_bias_f32
// CHECK-SAME:  (i64, i64, i64, i64, i64, i32, i32, i32, i32, f32, i32)

func.func @flash_bias(%Q: tensor<2x8x16xf32>, %K: tensor<2x8x16xf32>,
                      %V: tensor<2x8x16xf32>, %Bz: tensor<2x8x8xf32>) -> tensor<2x8x16xf32> {
  // CHECK-LABEL: func.func @flash_bias
  // CHECK:       call @tessera_apple_gpu_flash_attn_bias_f32
  // CHECK-NOT:   tessera.flash_attn
  %O = "tessera.flash_attn"(%Q, %K, %V, %Bz)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}>
        {causal = false, head_dim = 16 : i64}
        : (tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<2x8x8xf32>) -> tensor<2x8x16xf32>
  return %O : tensor<2x8x16xf32>
}

// Regression: the bias-free path still lowers to the plain MSL flash kernel.

// CHECK-LABEL: func.func @flash_nobias
// CHECK:       call @tessera_apple_gpu_flash_attn_f32
// CHECK-NOT:   call @tessera_apple_gpu_flash_attn_bias_f32
func.func @flash_nobias(%Q: tensor<2x8x16xf32>, %K: tensor<2x8x16xf32>,
                        %V: tensor<2x8x16xf32>) -> tensor<2x8x16xf32> {
  %O = "tessera.flash_attn"(%Q, %K, %V)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}>
        {causal = false, head_dim = 16 : i64}
        : (tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  return %O : tensor<2x8x16xf32>
}

// A broadcast (1, Sq, Sk) bias is legal IR (op verifier allows batch 1) but the
// (B, Sq, Sk) kernel cannot consume it; the lowering must NOT emit the bias
// symbol (it falls back to the reference path, which numpy-broadcasts). The op
// stays unlowered as tessera.flash_attn.

// CHECK-LABEL: func.func @flash_bias_broadcast
// CHECK-NOT:   call @tessera_apple_gpu_flash_attn_bias_f32
// CHECK:       tessera.flash_attn
func.func @flash_bias_broadcast(%Q: tensor<2x8x16xf32>, %K: tensor<2x8x16xf32>,
                                %V: tensor<2x8x16xf32>, %Bz: tensor<1x8x8xf32>) -> tensor<2x8x16xf32> {
  %O = "tessera.flash_attn"(%Q, %K, %V, %Bz)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}>
        {causal = false, head_dim = 16 : i64}
        : (tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<1x8x8xf32>) -> tensor<2x8x16xf32>
  return %O : tensor<2x8x16xf32>
}
