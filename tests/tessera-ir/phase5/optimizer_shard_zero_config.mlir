// RUN: tessera-opt --tessera-optimizer-shard --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-optimizer-shard="zero-stage=2 num-dp-ranks=8 dp-axis=data" --allow-unregistered-dialect %s | FileCheck %s

// TILE-LATENT-DEFECTS-2026-09-27. The module attribute below is, byte for
// byte, what Python emits for
//   ZeROConfig(stage=2, dp_axis="data", num_dp_ranks=8).to_ir_attr()
// (tests/unit/test_optimizer_shard.py pins that equality). The pass used to
// read `tessera.num_dp_ranks` / `tessera.dp_axis`, which nothing produces, so
// this configuration never arrived: the run with no options annotated
// partition_count = 1 and shard_axis = "dp". Now the configured values arrive,
// and options that restate them are accepted (second RUN).

// CHECK: module attributes
// CHECK-SAME: tessera_sr.dp_axis = "data"
// CHECK-SAME: tessera_sr.num_dp_ranks = 8 : i64
// CHECK-SAME: tessera_sr.zero_stage = 2 : i64
module attributes {tessera_sr.zero_config = {stage = 2, dp_axis = "data", num_ranks = 8}} {
  // CHECK-LABEL: func.func @adam_step
  func.func @adam_step(%m: memref<64xf32>) {
    // CHECK: "opt_test.adam_step"
    // CHECK-SAME: tessera_sr.partition_count = 8 : i64
    // CHECK-SAME: tessera_sr.shard_axis = "data"
    // CHECK-SAME: tessera_sr.zero_stage = 2 : i64
    "opt_test.adam_step"(%m) {tessera_sr.optimizer_state = "momentum"}
        : (memref<64xf32>) -> ()
    return
  }
}
