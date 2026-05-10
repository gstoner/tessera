// RUN: tessera-opt %s | FileCheck %s

// execution_roadmap.md, Phase F-MoR — Mixture of Recursions primitives
// parse-roundtrip lit fixture. Confirms ODS verifiers + assembly format
// for the three MoR ops.

// CHECK-LABEL: func.func @mor_pipeline
// CHECK:       tessera.mor_router
// CHECK:       tessera.mor_partition
// CHECK:       tessera.mor_scatter

func.func @mor_pipeline(%x: tensor<2x8x16xf32>,
                         %w_router: tensor<16x3xf32>,
                         %updated: tensor<2x8x16xf32>) -> tensor<2x8x16xf32> {
  %depth = "tessera.mor_router"(%x, %w_router) {max_depth = 3 : i64}
      : (tensor<2x8x16xf32>, tensor<16x3xf32>) -> tensor<2x8xi64>
  %mask = "tessera.mor_partition"(%x, %depth) {step = 1 : i64}
      : (tensor<2x8x16xf32>, tensor<2x8xi64>) -> tensor<2x8xi1>
  %out = "tessera.mor_scatter"(%x, %updated, %mask)
      : (tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<2x8xi1>) -> tensor<2x8x16xf32>
  return %out : tensor<2x8x16xf32>
}
