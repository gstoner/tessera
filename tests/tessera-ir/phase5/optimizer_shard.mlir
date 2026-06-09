// RUN: tessera-opt --tessera-optimizer-shard="zero-stage=2 num-dp-ranks=4 dp-axis=dp" --allow-unregistered-dialect %s | FileCheck %s

// Test: OptimizerShardPass annotates optimizer-state ops with ZeRO-2
// partitioning attributes.  The pass keys off the tessera_sr.optimizer_state
// attribute, not the carrier op type.
//
// 2026-06: un-XFAIL'd.  The carrier moved off the unregistered
// `tessera.optimizer.adam_step` op (which the `tessera` dialect doesn't define)
// onto a neutral unregistered-dialect op + --allow-unregistered-dialect; the
// pass annotation behaviour is unchanged.

module attributes {
  tessera.distributed_plan = {
    mesh = {"dp" = 4, "tp" = 2},
    total_ranks = 8
  }
} {

  // CHECK-LABEL: func.func @adam_step
  func.func @adam_step(
      %param:   memref<512x512xbf16>,
      %grad:    memref<512x512xbf16>,
      %moment1: memref<512x512xf32>,
      %moment2: memref<512x512xf32>
  ) -> memref<512x512xbf16> {

    // CHECK: tessera_sr.partition_count
    // CHECK-SAME: tessera_sr.sharded
    // CHECK-SAME: tessera_sr.zero_stage
    "opt_test.adam_step"(%param, %grad, %moment1, %moment2) {
      tessera_sr.optimizer_state = "momentum"
    } : (memref<512x512xbf16>, memref<512x512xbf16>,
         memref<512x512xf32>,   memref<512x512xf32>) -> ()

    return %param : memref<512x512xbf16>
  }
}
