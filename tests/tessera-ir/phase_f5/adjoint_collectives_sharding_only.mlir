// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s
// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s --check-prefix=PLAN
//
// Phase F5 — fallback path when the function is NOT effect-annotated (a
// pipeline that ran AutodiffPass but skipped EffectAnnotationPass). Without
// effect information the pass cannot gate, so it plans purely from
// weight_sharding and records the decision as "[sharding-only]" so the
// difference from the effect-gated path is visible downstream.

module {
  // CHECK: sym_name = "bwd_no_effect"
  func.func @bwd_no_effect(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>)
      -> (tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>)
      attributes {
        tessera.autodiff = "reverse",
        tessera.autodiff.arg_cotangents = ["%cotan_arg_0", "%cotan_arg_1"],
        tessera.weight_sharding = {arg_0 = "dp", arg_1 = "tp"}
      } {
    %out = "test.fwd"(%A, %B) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %dA = "test.cotan"(%A) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %dB = "test.cotan"(%B) : (tensor<8x16xf32>) -> tensor<8x16xf32>

    // CHECK: "tessera.collective.reduce_scatter"(%{{.*}}) {{.*}}axis = "dp"
    // CHECK: "tessera.collective.all_gather"(%{{.*}}) {{.*}}axis = "tp"
    func.return %out, %dA, %dB
        : tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>
  }
  // PLAN-DAG: tessera.adjoint_collective_plan = "reduce_scatter:dp [sharding-only]"
  // PLAN-DAG: tessera.adjoint_collective_plan = "all_gather:tp [sharding-only]"
}
