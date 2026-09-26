// REQUIRES: tessera-apple-backend
// RUN: not tessera-opt %s --split-input-file --tessera-tiling \
// RUN:   --tessera-apple-canonical-gemm-matmul2d --allow-unregistered-dialect 2>&1 \
// RUN:   | FileCheck %s
//
// APPLE-ACCUM-1 negative fixture for the Metal 4 matmul2d lane. The lane reads
// the program's numeric_policy.accum and refuses anything but fp32, because fp32
// is the only accumulator matmul2d implements: MPPTensorOpsMatMul2d.h lists
// half and bfloat *destinations*, but measured on the M1 Max (macOS 27.0) a
// half/bfloat destination is bit-exact with fp32 accumulation over the whole K
// rounded once to the destination -- an output rounding, not the reduced-
// precision accumulation accum = "fp16" declares. No view or matmul2d may be
// emitted for a refused GEMM.

// CHECK: APPLE_MATMUL2D_ACCUM_{{UNSUPPORTED}}: storage 'f8E4M3FN' x 'f8E4M3FN' declares accum="fp16", but the Metal 4 matmul2d lane (apple_gpu) accumulates in fp32 only
// CHECK-NOT: tessera_apple.gpu.matmul2d
func.func @fp8_gemm_declaring_fp16_accumulation(
    %a: tensor<64x128xf8E4M3FN>, %b: tensor<128x128xf8E4M3FN>) -> tensor<64x128xf32> {
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp8_e4m3", accum = "fp16"}}
      : (tensor<64x128xf8E4M3FN>, tensor<128x128xf8E4M3FN>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// -----

// bf16 accumulation is refused for the same measured reason.
// CHECK: APPLE_MATMUL2D_ACCUM_{{UNSUPPORTED}}: storage 'f16' x 'f8E5M2' declares accum="bf16"
func.func @weight_only_gemm_declaring_bf16_accumulation(
    %a: tensor<64x128xf16>, %b: tensor<128x128xf8E5M2>) -> tensor<64x128xf32> {
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "bf16"}}
      : (tensor<64x128xf16>, tensor<128x128xf8E5M2>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
