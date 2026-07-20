// RUN: tessera-opt --tessera-tile-to-x86='prefer-amx=true' %s  | FileCheck %s --check-prefix=AMX
// RUN: tessera-opt --tessera-tile-to-x86='prefer-amx=false' %s | FileCheck %s --check-prefix=AVX
// 2026-06: un-XFAIL'd.  Two MLIR-23 syntax drifts: bufferization.to_memref →
// to_buffer, and func.call now prints in pretty form as `call`.  Also the
// runtime fn declarations hoist to the top of the module (before the kernels),
// so the checks match in emitted order rather than via per-func LABEL anchors.

// (AMX path) matmul 16x32xbf16 @ 32x16xbf16 -> AMX kernel call
// AMX:        func.func private @tessera_x86_amx_gemm_bf16
// AMX:        func.func @tile_gemm_bf16
// AMX:        bufferization.to_buffer
// AMX:        memref.alloc
// AMX:        memref.extract_aligned_pointer_as_index
// AMX:        arith.index_cast
// AMX:        call @tessera_x86_amx_gemm_bf16
// AMX:        bufferization.to_tensor
// AMX-NOT:    tessera.matmul

// (AMX path) fused epilogue: matmul + bias + gelu -> GEMM + epilogue calls
// AMX:        func.func @fused_epilogue_gelu
// AMX:        call @tessera_x86_amx_gemm_bf16
// AMX:        call @tessera_x86_epilogue_bias_gelu_fp32

// (AVX path) same shape, AVX-512 kernel selected
// AVX:        func.func private @tessera_x86_avx512_gemm_bf16
// AVX:        func.func @tile_gemm_bf16
// AVX:        call @tessera_x86_avx512_gemm_bf16
// AVX-NOT:    tessera.matmul

module attributes {tessera.ir.version = "1.0"} {

  func.func @tile_gemm_bf16(
      %A: tensor<16x32xbf16>,
      %B: tensor<32x16xbf16>
  ) -> tensor<16x16xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<16x32xbf16>, tensor<32x16xbf16>)
                                    -> tensor<16x16xf32>
    return %C : tensor<16x16xf32>
  }

  func.func @fused_epilogue_gelu(
      %A:    tensor<16x32xbf16>,
      %B:    tensor<32x16xbf16>,
      %bias: tensor<16xf32>
  ) -> tensor<16x16xf32> {
    %C = "tessera.fused_epilogue"(%A, %B, %bias)
             {epilogue = 2 : i32, has_bias = true}
             : (tensor<16x32xbf16>, tensor<32x16xbf16>, tensor<16xf32>)
             -> tensor<16x16xf32>
    return %C : tensor<16x16xf32>
  }
}
