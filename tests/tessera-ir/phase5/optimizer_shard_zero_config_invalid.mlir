// RUN: not tessera-opt --tessera-optimizer-shard --allow-unregistered-dialect --split-input-file %s 2>&1 | FileCheck %s
// RUN: not tessera-opt --tessera-optimizer-shard="num-dp-ranks=4" --allow-unregistered-dialect %S/optimizer_shard_zero_config.mlir 2>&1 | FileCheck %s --check-prefix=OPTION

// TILE-LATENT-DEFECTS-2026-09-27. The ZeRO partition count, axis and stage are
// semantic keys (Decision #21a): the pass takes them from
// `tessera_sr.zero_config` or from explicit options and never defaults them.

// The old, never-produced module spellings no longer drive the pass, and with
// optimizer state to shard and no configuration it refuses instead of
// silently partitioning across 1 rank.
module attributes {tessera.num_dp_ranks = 8 : i64, tessera.dp_axis = "dp"} {
  func.func @unconfigured(%m: memref<64xf32>) {
    "opt_test.adam_step"(%m) {tessera_sr.optimizer_state = "momentum"}
        : (memref<64xf32>) -> ()
    return
  }
}
// CHECK: SR_ZERO_CONFIG_MISSING: optimizer state to shard but no ZeRO configuration
// CHECK-NOT: tessera_sr.partition_count

// -----

module attributes {tessera_sr.zero_config = {stage = 4, dp_axis = "dp", num_ranks = 8}} {
  func.func @bad_stage(%m: memref<64xf32>) {
    "opt_test.adam_step"(%m) {tessera_sr.optimizer_state = "momentum"}
        : (memref<64xf32>) -> ()
    return
  }
}
// CHECK: SR_ZERO_CONFIG_MALFORMED: tessera_sr.zero_config has stage = 4 outside {1, 2, 3}

// -----

module attributes {tessera_sr.zero_config = {stage = 2, dp_axis = "dp"}} {
  func.func @no_rank_count(%m: memref<64xf32>) {
    "opt_test.adam_step"(%m) {tessera_sr.optimizer_state = "momentum"}
        : (memref<64xf32>) -> ()
    return
  }
}
// CHECK: SR_ZERO_CONFIG_MALFORMED: tessera_sr.zero_config has no integer 'num_ranks'

// -----

// The mesh already states the axis size, so a disagreeing count is refused
// (derive, don't ask -- Decision #30).
module attributes {
  tessera.distributed_plan = {mesh = {"dp" = 4, "tp" = 2}, total_ranks = 8},
  tessera_sr.zero_config = {stage = 2, dp_axis = "dp", num_ranks = 8}
} {
  func.func @mesh_mismatch(%m: memref<64xf32>) {
    "opt_test.adam_step"(%m) {tessera_sr.optimizer_state = "momentum"}
        : (memref<64xf32>) -> ()
    return
  }
}
// CHECK: SR_ZERO_CONFIG_CONFLICT: ZeRO num_ranks = 8 but tessera.distributed_plan mesh axis 'dp' has 4 ranks

// -----

// A configured axis the mesh does not have names no partition: refused, not
// silently annotated (PR #867 review).
module attributes {
  tessera.distributed_plan = {mesh = {"dp" = 4, "tp" = 2}, total_ranks = 8},
  tessera_sr.zero_config = {stage = 2, dp_axis = "data", num_ranks = 4}
} {
  func.func @axis_not_in_mesh(%m: memref<64xf32>) {
    "opt_test.adam_step"(%m) {tessera_sr.optimizer_state = "momentum"}
        : (memref<64xf32>) -> ()
    return
  }
}
// CHECK: SR_ZERO_CONFIG_CONFLICT: ZeRO dp_axis 'data' is not a dimension of the tessera.distributed_plan mesh
// CHECK-NOT: tessera_sr.shard_axis = "data"

// An option may restate the configuration, never override it.
// OPTION: SR_ZERO_CONFIG_CONFLICT: --tessera-optimizer-shard num-dp-ranks=4 disagrees with tessera_sr.zero_config num-dp-ranks = 8
