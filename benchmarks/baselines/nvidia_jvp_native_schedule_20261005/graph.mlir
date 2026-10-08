module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120", tessera.autodiff.attention_jvp_contract = "{\22active\22:[true,true,true],\22causal\22:true,\22dims\22:[1,2,1,4,6,8,8],\22scale\22:0.35355338454246521,\22schema\22:1}"} {
  func.func @attention(%arg0: tensor<1x2x4x8xf32>, %arg1: tensor<1x1x6x8xf32>, %arg2: tensor<1x1x6x8xf32>) -> tensor<1x2x4x8xf32> attributes {tessera.autodiff = "forward", tessera.autodiff.jvp = @attention__jvp} {
    %0 = tessera.flash_attn %arg0, %arg1, %arg2 {causal = true, dropout_p = 0.000000e+00 : f64, head_dim = 8 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (tensor<1x2x4x8xf32>, tensor<1x1x6x8xf32>, tensor<1x1x6x8xf32>) -> tensor<1x2x4x8xf32>
    return %0 : tensor<1x2x4x8xf32>
  }
  func.func private @attention__jvp(%arg0: tensor<1x2x4x8xf32>, %arg1: tensor<1x1x6x8xf32>, %arg2: tensor<1x1x6x8xf32>, %arg3: tensor<1x2x4x8xf32>, %arg4: tensor<1x1x6x8xf32>, %arg5: tensor<1x1x6x8xf32>) -> (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) attributes {tessera.autodiff.forward = @attention, tessera.autodiff.role = "jvp"} {
    %0 = tessera.flash_attn %arg0, %arg1, %arg2 {causal = true, dropout_p = 0.000000e+00 : f64, head_dim = 8 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (tensor<1x2x4x8xf32>, tensor<1x1x6x8xf32>, tensor<1x1x6x8xf32>) -> tensor<1x2x4x8xf32>
    %output, %row_lse = tessera_attn.checkpoint_forward %arg0, %arg1, %arg2 {causal = true, scale = 0.353553385 : f32} : (tensor<1x2x4x8xf32>, tensor<1x1x6x8xf32>, tensor<1x1x6x8xf32>) -> (tensor<1x2x4x8xf32>, tensor<1x2x4xf32>)
    %1 = tessera_attn.checkpoint_jvp %arg0, %arg1, %arg2, %output, %row_lse, %arg3, %arg4, %arg5 {causal = true, scale = 0.353553385 : f32} : (tensor<1x2x4x8xf32>, tensor<1x1x6x8xf32>, tensor<1x1x6x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4xf32>, tensor<1x2x4x8xf32>, tensor<1x1x6x8xf32>, tensor<1x1x6x8xf32>) -> tensor<1x2x4x8xf32>
    return %0, %1 : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
  }
}

