// Sprint 11 — native sparse attention has one strict Apple GPU value envelope.
//
// RUN: %tessera_strict_opt -tessera-lower-to-apple_gpu-full %s | FileCheck %s

// CHECK-LABEL: func.func @native_sparse_value
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: abi = "msl_native_sparse_attn_f32"
// CHECK-SAME: block_size = 4
// CHECK-SAME: op_kind = "native_sparse_attn_fused"
// CHECK-SAME: status = "executable"
// CHECK-SAME: symbol = "tessera_apple_gpu_native_sparse_attn_f32"
// CHECK-SAME: top_k = 1
// CHECK-SAME: window_size = 4
// CHECK-NOT: tessera.native_sparse_attn_fused
// CHECK-NOT: ub.poison
func.func @native_sparse_value(
    %q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>,
    %v: tensor<1x2x8x4xf32>, %gate: tensor<1x2x8x2xf32>)
    -> tensor<1x2x8x4xf32> {
  %0 = tessera.native_sparse_attn_fused %q, %k, %v, %gate
       {window_size = 4 : i64, block_size = 4 : i64, top_k = 1 : i64,
        causal = true}
       : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
          tensor<1x2x8x4xf32>, tensor<1x2x8x2xf32>)
         -> tensor<1x2x8x4xf32>
  return %0 : tensor<1x2x8x4xf32>
}
