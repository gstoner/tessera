// RUN: tessera-opt --tessera-activation-rematerialization %s | FileCheck %s
//
// A function-level budget is the production-pipeline contract: the same pass
// that follows AutodiffPass selects long-lived pure activations, then sinks
// their recomputation.  No frontend tessera.recompute marker is required.

module {
  // CHECK: func.func @budget_selected
  // CHECK-SAME: tessera.remat_auto_selected = 2
  // CHECK-SAME: tessera.remat_budget_mb = 1
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

  // CHECK-NOT: tessera.recompute
}
