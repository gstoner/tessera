// RUN: %tnv %s | FileCheck %s
//
// Typed, hardware-free promotion artifacts for the consumer-Blackwell lanes.
// Parsing this file without --allow-unregistered-dialect is the Target IR
// verifier rung.  Runtime promotion remains independently gated by the NVIDIA
// execute/compare and performance tests named below.

module {
  func.func @sm120_differentiation(%a: tensor<16x64xf16>,
                                   %b: tensor<64x8xf16>,
                                   %q: tensor<16x64xf16>,
                                   %k: tensor<16x64xf16>,
                                   %v: tensor<16x64xf16>,
                                   %x: tensor<128xf32>,
                                   %pa: tensor<16x64xi4>,
                                   %pb: tensor<64x8xi4>,
                                   %sfa: tensor<16x4xf8E4M3FN>,
                                   %sfb: tensor<4x8xf8E4M3FN>) {
    // CHECK: tessera_nvidia.mma_fused
    // CHECK-SAME: arch = "sm_120"
    // CHECK-SAME: epilogue = "bias+gelu"
    %gemm = tessera_nvidia.mma_fused %a, %b
        {arch = "sm_120", shape = "m16n8k16", dtype_ab = "f16",
         dtype_c = "f32", accum = "f32", epilogue = "bias+gelu"}
        : (tensor<16x64xf16>, tensor<64x8xf16>) -> tensor<16x8xf32>

    // CHECK: tessera_nvidia.mma_attention
    // CHECK-SAME: accum = "f32"
    // CHECK-SAME: mask = "causal_or_none"
    %attn = tessera_nvidia.mma_attention %q, %k, %v
        {arch = "sm_120", shape = "m16n8k16", dtype_ab = "f16",
         dtype_c = "f32", accum = "f32", mask = "causal_or_none"}
        : (tensor<16x64xf16>, tensor<16x64xf16>, tensor<16x64xf16>)
          -> tensor<16x64xf32>

    // These are storage conversions with f32 compute today.  The artifact does
    // not claim FP8/FP6 tensor-core MMA support.
    // CHECK: tessera_nvidia.fpquant
    // CHECK-SAME: format = "e4m3"
    %fp8 = tessera_nvidia.fpquant %x
        {arch = "sm_120", format = "e4m3", source = "quantize",
         dtype_ab = "f32", dtype_c = "f32", accum = "f32"}
        : (tensor<128xf32>) -> tensor<128xf32>
    // CHECK: tessera_nvidia.fpquant
    // CHECK-SAME: format = "e3m2"
    %fp6 = tessera_nvidia.fpquant %x
        {arch = "sm_120", format = "e3m2", source = "quantize",
         dtype_ab = "f32", dtype_c = "f32", accum = "f32"}
        : (tensor<128xf32>) -> tensor<128xf32>

    // CHECK: tessera_nvidia.nvfp4_block_scale_mma
    // CHECK-SAME: arch = "sm_120a"
    // CHECK-SAME: block_scaled = true
    %fp4 = tessera_nvidia.nvfp4_block_scale_mma %pa, %pb, %sfa, %sfb
        {arch = "sm_120a", shape = "m16n8k64", format = "e2m1+ue4m3",
         dtype_ab = "e2m1", dtype_c = "f32", accum = "f32",
         block_scaled = true}
        : (tensor<16x64xi4>, tensor<64x8xi4>, tensor<16x4xf8E4M3FN>,
           tensor<4x8xf8E4M3FN>) -> tensor<16x8xf32>
    return
  }
}
