// RUN: ts-clifford-opt --tessera-clifford-expand-product-table %s | FileCheck %s
//
// W6.4 batched products (2026-09-16): a `[..., dim]` operand pair lowers to
// the same compile-time-known Cayley contraction inside an scf.for nest over
// the leading axes, carrying the result tensor as an iter_arg. Every
// coefficient is written (pruned grades as zero), and no geo_product survives.

module {
  // CHECK-LABEL: func.func @cl30_batched
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<32x8xf32>
  // CHECK: %[[R:.*]] = scf.for %[[B:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<32x8xf32>)
  // CHECK: tensor.extract %{{.*}}[%[[B]], %{{.*}}] : tensor<32x8xf32>
  // CHECK: arith.mulf
  // CHECK: tensor.insert %{{.*}} into %[[ACC]][%[[B]], %{{.*}}] : tensor<32x8xf32>
  // CHECK: scf.yield %{{.*}} : tensor<32x8xf32>
  // CHECK: return %[[R]] : tensor<32x8xf32>
  // CHECK-NOT: tessera_clifford.geo_product
  func.func @cl30_batched(
      %a : tensor<32x8xf32>, %b : tensor<32x8xf32>) -> tensor<32x8xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    return %r : tensor<32x8xf32>
  }

  // Rank 3 nests two loops; the grade restriction still prunes the table
  // (grade-2 output of Cl(3,0): the scalar coefficient is written as zero,
  // never computed).
  // CHECK-LABEL: func.func @cl30_batched_rank3_grade2
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : tensor<4x5x8xf32>
  // CHECK-NOT: tessera_clifford.geo_product
  func.func @cl30_batched_rank3_grade2(
      %a : tensor<4x5x8xf32>, %b : tensor<4x5x8xf32>) -> tensor<4x5x8xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32", tessera.clifford.output_grades = [2] }
        : (tensor<4x5x8xf32>, tensor<4x5x8xf32>) -> tensor<4x5x8xf32>
    return %r : tensor<4x5x8xf32>
  }
}
