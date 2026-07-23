// RUN: tessera-opt --tessera-autodiff-pipeline --verify-each=false %s | FileCheck %s
//
// Production memory planning over a real compiler-owned backward graph.
// Softmax backward needs the forward probabilities. With a 1 MiB function
// budget, the 4 MiB activation is selected globally and recomputed next to its
// backward consumer; the original remains for the public function result.

module {
  // CHECK-LABEL: func.func @softmax_train
  // CHECK-SAME: tessera.remat_auto_selected = 1
  // CHECK-SAME: tessera.remat_budget_mb = 1
  // Two backward consumers share the activation contract and each receives a
  // short-lived recomputation under the strict memory budget.
  // CHECK-SAME: tessera.rematerialized = 2
  func.func @softmax_train(%x: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32>
      attributes {
        tessera.autodiff = "reverse",
        tessera.remat_budget_mb = 1 : i32
      } {
    // The public forward value.
    // CHECK: %[[FORWARD:.+]] = tessera.softmax %arg0
    %y = "tessera.softmax"(%x) {axis = -1 : i64} :
        (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    func.return %y : tensor<1024x1024xf32>
  }

  // The selected clone is adjacent to the native softmax VJP graph.
  // CHECK: %[[RECOMPUTED:.+]] = tessera.softmax %arg0
  // CHECK: tessera.mul %{{.*}}, %[[RECOMPUTED]]
  // CHECK-NOT: tessera.recompute
}
