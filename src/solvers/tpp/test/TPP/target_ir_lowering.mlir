// RUN: tessera-opt %s -tpp-halo-infer -tpp-distribute-halo -lower-tpp-to-target-ir | FileCheck %s
//
// LowerTPPToTargetIR annotates every TPP op with a hardware-free Target-IR
// call symbol chosen by the module's tessera.target.  Here backend = amd, so
// grad/halo.exchange/bc.enforce route to the *_amd primitives, and bc.enforce
// keeps its historical masked-store marker.

module attributes {tessera.target = "amd", tessera.mesh.axes = ["x"]} {
  func.func @sw(%h: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %g = "tpp.grad"(%h) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %b = "tpp.bc.enforce"(%g) { bc = #tpp.bc<"periodic"> } : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %b : tensor<64x64xf32>
  }
}

// CHECK: tpp.halo.exchange
// CHECK-SAME: tessera.target_ir.call = "ts_halo_exchange_amd"
// CHECK: tpp.grad
// CHECK-SAME: tessera.target_ir.arbiter_op = "tpp_stencil"
// CHECK-SAME: tessera.target_ir.call = "ts_stencil_grad_amd"
// CHECK: tpp.bc.enforce
// CHECK-SAME: lowered.bc.masked
// CHECK-SAME: tessera.target_ir.call = "ts_bc_enforce_amd"
