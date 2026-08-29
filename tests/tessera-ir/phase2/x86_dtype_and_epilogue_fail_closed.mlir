// RUN: not tessera-opt --tessera-tile-to-x86 %s 2>&1 | FileCheck %s
//
// Two silent-wrong-answer paths in the x86 lowering, both now fail closed.
//
// 1. f16 operands. The only GEMM symbols the C shim exports are
//    `tessera_x86_{amx,avx512}_gemm_bf16`; their ABI carries no dtype selector
//    and the kernel decodes each uint16 as bf16. Accepting f16 reinterpreted a
//    5/10 exponent/mantissa split as 8/7 — f16 1.0 (0x3C00) read as bf16 is
//    ~0.0117 — with every IR type self-consistent, so no verifier caught it.
//
// 2. An epilogue with no matching kernel. `..._epilogue_bias_fp32` applies a
//    bias and NO activation, so routing RELU there dropped the activation and
//    replaced the op with a bias-only result.
//
// Decision #21: an unsupported lowering names the op and the target rather
// than falling through to a wrong-but-plausible result.

// Pattern application order is not fixed, so match both diagnostics unordered.
// CHECK-DAG: error: x86 GEMM lowering has no f16 kernel
// CHECK-DAG: error: x86 fused-epilogue lowering supports epilogue none|gelu

func.func @f16_matmul(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
  %0 = "tessera.matmul"(%a, %b) {transposeA = false, transposeB = false}
      : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

func.func @relu_epilogue(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>,
                         %bias: tensor<64xf32>) -> tensor<64x64xf32> {
  %0 = "tessera.fused_epilogue"(%a, %b, %bias) {epilogue = 1 : i32, has_bias = true}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
