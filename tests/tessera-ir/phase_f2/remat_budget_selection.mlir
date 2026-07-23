// RUN: tessera-opt --tessera-activation-rematerialization %s | FileCheck %s
//
// A function-level budget is the production-pipeline contract: the same pass
// that follows AutodiffPass selects long-lived pure activations, then sinks
// their recomputation.  No frontend tessera.recompute marker is required.

module {
  // CHECK: func.func @budget_selected
  // CHECK-SAME: tessera.remat_auto_selected = 2
  // CHECK-SAME: tessera.remat_budget_mb = 1
  // CHECK-SAME: tessera.remat_peak_after_bytes = 0
  // CHECK-SAME: tessera.remat_peak_before_bytes = 8388608
  // CHECK-SAME: tessera.remat_selected_cost_ns = 2097152
  // CHECK-SAME: tessera.rematerialized = 2
  func.func @budget_selected(%x: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32>
      attributes {tessera.remat_budget_mb = 1 : i32} {
    %square = arith.mulf %x, %x : tensor<1024x1024xf32>
    %neg = arith.negf %x : tensor<1024x1024xf32>
    // CHECK: arith.negf
    // CHECK-NEXT: arith.mulf
    // CHECK-NEXT: arith.addf
    %sum = arith.addf %square, %neg : tensor<1024x1024xf32>
    return %sum : tensor<1024x1024xf32>
  }

  // Equal-size/lifetime activations use measured recompute latency as the
  // deciding signal. Only the cheap producer is needed to meet this budget.
  // CHECK-LABEL: func.func @measured_cost_prefers_cheap
  // CHECK-SAME: tessera.remat_auto_selected = 1
  // CHECK-SAME: tessera.remat_peak_after_bytes = 1048576
  // CHECK-SAME: tessera.remat_peak_before_bytes = 2097152
  // CHECK-SAME: tessera.remat_selected_cost_ns = 25
  func.func @measured_cost_prefers_cheap(
      %x: tensor<512x512xf32>) -> tensor<512x512xf32>
      attributes {tessera.remat_budget_mb = 1 : i32} {
    %expensive = arith.mulf %x, %x
        {tessera.remat_cost_ns = 900 : i64} : tensor<512x512xf32>
    %cheap = arith.negf %x
        {tessera.remat_cost_ns = 25 : i64} : tensor<512x512xf32>
    // CHECK: arith.negf
    // CHECK-NEXT: arith.addf
    %sum = arith.addf %expensive, %cheap : tensor<512x512xf32>
    return %sum : tensor<512x512xf32>
  }

  // CHECK-NOT: tessera.recompute
}
