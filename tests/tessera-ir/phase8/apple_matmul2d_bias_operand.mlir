// REQUIRES: tessera-apple-backend
// RUN: tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d \
// RUN:   --tessera-apple-matmul2d-fuse-epilogue --allow-unregistered-dialect \
// RUN:   | FileCheck %s --check-prefix=APPLE
//
// The re-applied epilogue is the form the Apple matmul2d fusion consumes, so
// the bias-operand matmul now reaches `gpu.matmul2d_epilogue` end to end.
// APPLE-LABEL: func.func @matmul_bias_f16
// APPLE: tessera_apple.gpu.matmul2d_epilogue %{{.*}}, %{{.*}} bias %arg2 : tensor<96xf32>
// APPLE-SAME: act = "none"
func.func @matmul_bias_f16(%a: tensor<64x128xf16>, %b: tensor<128x96xf16>, %bias: tensor<96xf32>) -> tensor<64x96xf32> {
  %0 = tessera.matmul %a, %b, %bias {bias = "row"} : (tensor<64x128xf16>, tensor<128x96xf16>, tensor<96xf32>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}
