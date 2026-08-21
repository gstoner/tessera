// W4-PRODUCT-1 — the state-machine kernel-gen FAILS CLOSED (Decision #21):
// a machine whose vocabulary exceeds the per-thread elementwise contract —
// here an op outside the admitted vocabulary in a state slot update — is
// declined with a remark naming the reason, and no kernel is emitted. The
// function is hand-written in the already-structurized form (exec attr
// present) so the negative targets exactly this pass.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --generate-rocm-state-machine-kernel \
// RUN:   --verify-diagnostics --allow-unregistered-dialect | FileCheck %s

module {
  // expected-remark @+1 {{not lowered to a ROCm state-machine kernel: unsupported op testx.opaque}}
  func.func @machine_with_matmul(%x: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%s = %x)
        -> (tensor<8xf32>) {
      %n = "testx.opaque"(%s, %s) :
          (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
      scf.yield %n : tensor<8xf32>
    } {tessera.structured_cfg.execution = "bounded_state_machine_v1",
       tessera.structured_cfg.digest = "6666666666666666666666666666666666666666666666666666666666666666",
       tessera.structured_cfg.max_steps = 8 : i64}
    return %r : tensor<8xf32>
  }
}

// CHECK-NOT: gpu.module
// CHECK-NOT: gpu.func
// CHECK: func.func @machine_with_matmul
// CHECK-NOT: tessera.rocm_kernel
