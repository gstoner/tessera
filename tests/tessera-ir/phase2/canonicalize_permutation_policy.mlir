// RUN: tessera-opt %s --tessera-canonicalize | FileCheck %s
// RUN: tessera-opt %s --canonicalize | FileCheck %s --check-prefix=GEN

// Equal shapes do not make two three-cycles cancel.
// CHECK-LABEL: func.func @compose
// CHECK: tessera.transpose
// CHECK-SAME: permutation = array<i64: 2, 0, 1>
// CHECK-NOT: tessera.transpose
// GEN-LABEL: func.func @compose
// GEN: permutation = array<i64: 2, 0, 1>
func.func @compose(%x: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %a = tessera.transpose %x {permutation = array<i64: 1, 2, 0>} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  %b = tessera.transpose %a {permutation = array<i64: 1, 2, 0>} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  return %b : tensor<2x2x2xf32>
}
// CHECK-LABEL: func.func @inverse
// CHECK-NOT: tessera.transpose
// CHECK: return %arg0
// GEN-LABEL: func.func @inverse
// GEN-NOT: tessera.transpose
// GEN: return %arg0
func.func @inverse(%x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %a = tessera.transpose %x {permutation = array<i64: 1, 2, 0>} : (tensor<2x3x4xf32>) -> tensor<3x4x2xf32>
  %b = tessera.transpose %a {permutation = array<i64: 2, 0, 1>} : (tensor<3x4x2xf32>) -> tensor<2x3x4xf32>
  return %b : tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @identity_matmul
// CHECK-NOT: transposeA = true
// CHECK: tessera.matmul %arg0, %arg1
// GEN-LABEL: func.func @identity_matmul
// GEN-NOT: transposeA = true
// GEN: tessera.matmul %arg0, %arg1
func.func @identity_matmul(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %a = tessera.transpose %x {permutation = array<i64: 0, 1>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %b = tessera.matmul %a, %y : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %b : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @cast_contract
// CHECK: tessera.cast
// CHECK-SAME: tessera.proof = "keep"
// GEN-LABEL: func.func @cast_contract
// GEN: tessera.cast
func.func @cast_contract(%x: tensor<2xf32>) -> tensor<2xf32> {
  %a = tessera.cast %x {tessera.proof = "keep"} : (tensor<2xf32>) -> tensor<2xf32>
  return %a : tensor<2xf32>
}
// CHECK-LABEL: func.func @fusion_contract
// CHECK-NOT: tessera.fused_epilogue
// CHECK: tessera.matmul
// CHECK-SAME: tile_k = 8
// CHECK: tessera.gelu
// GEN-LABEL: func.func @fusion_contract
// GEN: tile_k = 8
func.func @fusion_contract(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>, %bias: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %a = tessera.matmul %x, %y {tile_k = 8 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %b = tessera.add %a, %bias : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %c = tessera.gelu %b : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @cast_policy
// CHECK: tessera.cast
// CHECK-SAME: numeric_policy
// GEN-LABEL: func.func @cast_policy
// GEN: tessera.cast
// GEN-SAME: numeric_policy
func.func @cast_policy(%x: tensor<2xf32>) -> tensor<2xf32> {
  %a = tessera.cast %x {numeric_policy = {storage = "fp32", accum = "fp32"}} : (tensor<2xf32>) -> tensor<2xf32>
  return %a : tensor<2xf32>
}
// CHECK-LABEL: func.func @bias_operand
// CHECK: tessera.matmul
// CHECK-SAME: %arg2
// GEN-LABEL: func.func @bias_operand
// GEN: tessera.matmul %arg0, %arg1, %arg2
// GEN-SAME: transposeA = true
func.func @bias_operand(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %bias: tensor<4xf32>) -> tensor<2x4xf32> {
  %a = tessera.transpose %x : (tensor<3x2xf32>) -> tensor<2x3xf32>
  %b = tessera.matmul %a, %y, %bias {bias = "row"} : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %b : tensor<2x4xf32>
}
// CHECK-LABEL: func.func @conv_policy
// CHECK: tessera.conv2d_nhwc
// CHECK: tessera.relu
// CHECK-SAME: numeric_policy
// GEN-LABEL: func.func @conv_policy
// GEN: tessera.relu
// GEN-SAME: numeric_policy
func.func @conv_policy(%x: tensor<1x8x8x4xf32>, %w: tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32> {
  %c = "tessera.conv2d_nhwc"(%x,%w) {strides = [1,1], dilations = [1,1]} : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32>
  %r = tessera.relu %c {numeric_policy = {storage = "fp32", accum = "fp32"}} : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}
