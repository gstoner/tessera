// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s
// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s --check-prefix=PLAN
// RUN: tessera-opt --tessera-adjoint-collective-insertion --verify-each=false %s | FileCheck %s --check-prefix=NONE
//
// Phase F5 — effect-aware gating skips non-memory arguments.
//
// Both args are dp-sharded, but only arg_0 carries a memory-class effect.
// arg_1 is a read-only activation ("pure"): its cotangent must NOT get a
// gradient collective — synchronising it would double-count. The pass records
// the deliberate skip in the plan attribute.

module {
  // CHECK: sym_name = "bwd_gated"
  func.func @bwd_gated(%W: tensor<4x8xf32> {tessera.effect = "memory"},
                       %X: tensor<8x16xf32> {tessera.effect = "pure"})
      -> (tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>)
      attributes {
        tessera.autodiff = "reverse",
        tessera.autodiff.arg_cotangents = ["%cotan_arg_0", "%cotan_arg_1"],
        tessera.weight_sharding = {arg_0 = "dp", arg_1 = "dp"}
      } {
    %out = "test.fwd"(%W, %X) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %dW = "test.cotan"(%W) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %dX = "test.cotan"(%X) : (tensor<8x16xf32>) -> tensor<8x16xf32>

    // Exactly one collective — for the memory-class weight. The pure
    // activation's cotangent gets none (asserted by the NONE prefix below).
    // CHECK: "tessera.collective.reduce_scatter"
    func.return %out, %dW, %dX
        : tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>
  }
  // PLAN-DAG: tessera.adjoint_collective_plan = "reduce_scatter:dp [effect-gated]"
  // PLAN-DAG: tessera.adjoint_collective_plan = "none:non-memory-effect=pure"

  // No all_gather / all_reduce anywhere (the pure activation got no collective).
  // NONE-NOT: all_gather
  // NONE-NOT: all_reduce
}
