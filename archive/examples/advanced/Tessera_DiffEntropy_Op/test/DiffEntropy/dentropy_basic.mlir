// RUN: tessera-dentropy-opt %s | FileCheck %s
// Simple smoke test for both ops (shapes illustrative)

// CHECK-LABEL: func @test_dentropy_range
func.func @test_dentropy_range(%points: memref<1x8x3xf32>, %anchors: memref<1x4x3xf32>) -> memref<f32> {
  %h = "tessera.diffentropy.range_entropy_soft"(%points, %anchors) {alpha = 0.5 : f64, range_family = "balls", reduction = "sum"} : (memref<1x8x3xf32>, memref<1x4x3xf32>) -> memref<f32>
  // CHECK: "tessera.diffentropy.range_entropy_soft"
  return %h : memref<f32>
}

// CHECK-LABEL: func @test_dentropy_attn
func.func @test_dentropy_attn(%attn: memref<2x8x16x16xf32>) -> memref<f32> {
  %h = "tessera.diffentropy.attn_row_entropy"(%attn) {tau = 1.0 : f64, mode = "scores", reduction = "mean"} : (memref<2x8x16x16xf32>) -> memref<f32>
  // CHECK: "tessera.diffentropy.attn_row_entropy"
  return %h : memref<f32>
}
