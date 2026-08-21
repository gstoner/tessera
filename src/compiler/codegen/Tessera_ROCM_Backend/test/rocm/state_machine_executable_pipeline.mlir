// PR #605 review (P1) — the state-machine family is wired into the CANONICAL
// registered executable pipeline: `family=control_state_machine` runs the
// kernel generator inside `tessera-rocm-executable` and serializes the
// per-thread machine to a gpu.binary — normal ROCm binary compilation emits
// these kernels; no hand-assembled pass list is required.
//
// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=control_state_machine input=tile output=binary arch=gfx1151})' --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @machine(%x: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%s = %x)
        -> (tensor<8xf32>) {
      %n = "tessera.tanh"(%s) : (tensor<8xf32>) -> tensor<8xf32>
      scf.yield %n : tensor<8xf32>
    } {tessera.structured_cfg.execution = "bounded_state_machine_v1",
       tessera.structured_cfg.digest = "8888888888888888888888888888888888888888888888888888888888888888",
       tessera.structured_cfg.max_steps = 8 : i64}
    return %r : tensor<8xf32>
  }
}

// CHECK-DAG: tessera.pipeline.family = "control_state_machine"
// CHECK-DAG: tessera.pipeline.schema = "tessera.executable_pipeline.v1"
// CHECK: func.func @machine
// CHECK-SAME: tessera.rocm_kernel = "tessera_state_machine_machine"
// CHECK: gpu.binary @tessera_state_machine_machine_mod
