// W4-PRODUCT-1 (PR #605 review, P2) — INTERIOR values may be rank-1 tensors
// of i1 over the common size: a per-element data-dependent state update
// (cmpf over the data slots feeding select) scalarizes to i1/f32 rather than
// leaving a tensor<Nxi1> result on an op with scalar operands. On-device
// execution of this machine is proven by
// tests/unit/test_rocm_state_machine_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --generate-rocm-state-machine-kernel | FileCheck %s

module {
  func.func @select_machine(%x: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%s = %x)
        -> (tensor<8xf32>) {
      %zero = arith.constant dense<0.0> : tensor<8xf32>
      %p = arith.cmpf ogt, %s, %zero : tensor<8xf32>
      %t = "tessera.tanh"(%s) : (tensor<8xf32>) -> tensor<8xf32>
      %n = arith.select %p, %t, %s : tensor<8xi1>, tensor<8xf32>
      scf.yield %n : tensor<8xf32>
    } {tessera.structured_cfg.execution = "bounded_state_machine_v1",
       tessera.structured_cfg.digest = "7777777777777777777777777777777777777777777777777777777777777777",
       tessera.structured_cfg.max_steps = 4 : i64}
    return %r : tensor<8xf32>
  }
}

// CHECK: gpu.func @tessera_state_machine_select_machine(
// CHECK-SAME: tessera.structured_cfg.digest
// The comparison and selection are SCALAR inside the kernel — tensor<8xi1>
// became i1, tensor<8xf32> became f32.
// CHECK: %[[P:.*]] = arith.cmpf ogt, %{{.*}} : f32
// CHECK: %[[T:.*]] = math.tanh %{{.*}} : f32
// CHECK: arith.select %[[P]], %[[T]], %{{.*}} : f32
// CHECK: gpu.return
