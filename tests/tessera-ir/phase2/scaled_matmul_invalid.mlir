// RUN: not tessera-opt %s 2>&1 | FileCheck %s

// The negative case is the whole point: a scale tensor with the wrong number of
// blocks is NUMERICALLY SILENT without this check -- the GEMM still runs, with
// the wrong factor on every block but one. K = 512 over blocks of 128 needs an
// axis of 4; this one has 3.
// CHECK: 'tessera.scaled_matmul' op lhs_scale has no axis of extent 4
// CHECK-SAME: cannot carry one scale per K block
func.func @wrong_block_count(%a: tensor<128x512xf8E4M3FN>, %b: tensor<512x128xf8E4M3FN>,
                             %sa: tensor<128x3xf32>, %sb: tensor<4x128xf32>) -> tensor<128x128xf32> {
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb)
       {scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}}
       : (tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>, tensor<128x3xf32>, tensor<4x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
