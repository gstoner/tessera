// REQUIRES: tessera-apple-backend
// RUN: not tessera-opt %s --split-input-file --tessera-tiling \
// RUN:   --tessera-apple-canonical-gemm --allow-unregistered-dialect 2>&1 \
// RUN:   | FileCheck %s
//
// APPLE-ACCUM-1 negative fixture for the canonical-GEMM simdgroup route. The
// route reads its accumulator from IR -- the canonical nest's loop-carried
// accumulator type -- and cross-checks a declared numeric_policy.accum. A
// program that declares fp16 accumulation on a nest the shared tiler built with
// an fp32 accumulator does not compute what it declared, so the route refuses
// rather than dispatching the fp32 kernel under an fp16 label.

// CHECK: APPLE_CANONICAL_GEMM_ACCUM_{{UNSUPPORTED}}: numeric_policy.accum="fp16" disagrees with the canonical reduction's loop-carried 'f32' accumulator (storage 'f16', target apple_gpu)
// CHECK-NOT: tile_simdgroup_gemm
func.func @declared_fp16_on_an_fp32_nest(
    %a: tensor<32x32xf16>, %b: tensor<32x32xf16>) -> tensor<32x32xf32> {
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "fp16"}}
      : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
