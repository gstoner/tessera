// RUN: tessera-opt --tessera-control-flow-to-scf %s | FileCheck %s --check-prefix=LOWER
// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-control-flow-to-scf,tessera-autodiff-forward)' %s | FileCheck %s --check-prefix=JVP
// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-control-flow-to-scf,tessera-autodiff-paired)' %s | FileCheck %s --check-prefix=VJP
//
// W4 control_scan product boundary.  Lowering inlines the body and makes the
// stream gather/stack explicit. Forward and reverse products therefore use
// the ordinary SCF region interfaces plus exact tensor slice derivatives.

module {
  func.func @scan_step(%carry: tensor<4xf32>, %xt: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %next = "tessera.add"(%carry, %xt) :
        (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %next, %next : tensor<4xf32>, tensor<4xf32>
  }

  func.func @scan_forward(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> (tensor<4xf32>, tensor<3x4xf32>) attributes {
        tessera.autodiff = "forward"} {
    %carry, %ys = "tessera.control_scan"(%init, %xs) {
      body = @scan_step, carry_arg_index = 0 : i64, trip = 3 : i64
    } : (tensor<4xf32>, tensor<3x4xf32>) ->
        (tensor<4xf32>, tensor<3x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<3x4xf32>
  }

  func.func @scan_reverse(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
      -> (tensor<4xf32>, tensor<3x4xf32>) attributes {
        tessera.autodiff = "reverse"} {
    %carry, %ys = "tessera.control_scan"(%init, %xs) {
      body = @scan_step, carry_arg_index = 0 : i64, trip = 3 : i64
    } : (tensor<4xf32>, tensor<3x4xf32>) ->
        (tensor<4xf32>, tensor<3x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<3x4xf32>
  }
}

// LOWER-LABEL: func.func @scan_forward
// LOWER: %[[EMPTY:.+]] = tensor.empty() : tensor<3x4xf32>
// LOWER: %[[SCAN:.+]]:2 = scf.for
// LOWER: tensor.extract_slice
// LOWER: tessera.add
// LOWER: tensor.insert_slice
// LOWER: scf.yield
// LOWER: } {tessera.autodiff.checkpoint_policy = "recompute_all"}
// LOWER-NOT: tessera.control_scan

// JVP-LABEL: func.func private @scan_forward__jvp
// JVP-COUNT-1: scf.for
// JVP: tensor.extract_slice
// JVP: tensor.extract_slice
// JVP: tessera.add
// JVP: tensor.insert_slice
// JVP: tensor.insert_slice
// JVP: return

// VJP-LABEL: func.func @scan_reverse__bwd
// VJP: scf.for
// VJP: scf.for
// VJP: tensor.extract_slice
// VJP: tensor.insert_slice
// VJP: return
