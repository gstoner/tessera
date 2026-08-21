// PR #606 review — the canonical state-machine route FAILS CLOSED:
//  (P1) a requested family that cannot realize every machine (here: a
//       machine with no structured-CFG digest) fails the PIPELINE instead
//       of sailing through gpu-module-to-binary and reporting success with
//       no gpu.binary;
//  (P2) output=target is rejected up front — this family has no
//       tessera_rocm.* Target-IR boundary, so a target artifact would be
//       the untouched host program relabeled.
//
// RUN: not %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=control_state_machine input=tile output=binary arch=gfx1151})' --allow-unregistered-dialect %s 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: not %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=control_state_machine input=tile output=target arch=gfx1151})' --allow-unregistered-dialect %s 2>&1 | FileCheck %s --check-prefix=TARGET

module {
  func.func @no_digest(%x: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%s = %x)
        -> (tensor<8xf32>) {
      %n = "tessera.tanh"(%s) : (tensor<8xf32>) -> tensor<8xf32>
      scf.yield %n : tensor<8xf32>
    } {tessera.structured_cfg.execution = "bounded_state_machine_v1",
       tessera.structured_cfg.max_steps = 8 : i64}
    return %r : tensor<8xf32>
  }
}

// STRICT: could not realize every bounded state machine as a device kernel
// TARGET: family 'control_state_machine' has no Target-IR boundary; only output=binary is supported
