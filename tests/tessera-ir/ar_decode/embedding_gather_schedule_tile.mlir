// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE
//
// AR-DECODE-PROGRAM-1 S1c: an sm_120 embedding lookup (tessera.gather of
// table rows by i32 tokens) descends Graph -> Schedule -> Tile with a
// content-addressed native contract. The contract names the out-of-range rule
// (the kernel checks every token on the device and refuses through a status
// word) and the physical schedule (one 128-thread CTA per token row).

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @embed(%table: tensor<512x64xf32>, %tokens: tensor<3xi32>) -> tensor<3x64xf32>
      attributes {tessera.bindings = ["embed", "token", "x"]} {
    %x = tessera.gather %table, %tokens {axis = 0 : i64} : (tensor<512x64xf32>, tensor<3xi32>) -> tensor<3x64xf32>
    return %x : tensor<3x64xf32>
  }
}

// SCHED-LABEL: func.func @embed
// SCHED: tessera.gather {{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHED: schedule.embedding_gather
// SCHED-SAME: artifact_hash = "[[HASH]]"
// SCHED-SAME: bindings = ["embed", "token", "x"]
// SCHED-SAME: index_bounds = "kernel_checked_status_word"
// SCHED-SAME: index_storage = "i32"
// SCHED-SAME: schedule = "cta_128_per_token_row"
// SCHED-SAME: shape = array<i64: 512, 64, 3>
// SCHED-SAME: storage = "f32"

// TILE: llvm.func @tessera_tile_embedding_gather_f32_{{[0-9a-f]+}}(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
// TILE-SAME: nvvm.kernel
// TILE-SAME: nvvm.reqntid = array<i32: 128, 1, 1>
// TILE: tile.embedding_gather_kernel
// TILE-SAME: index_bounds = "kernel_checked_status_word", index_storage = "i32", schedule = "cta_128_per_token_row", storage = "f32"
// TILE-NOT: func.func @embed

// -----

// The ODS default axis (0) is the same contract as an explicit one.
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @default_axis(%table: tensor<7x5xf32>, %tokens: tensor<1xi32>) -> tensor<1x5xf32>
      attributes {tessera.bindings = ["t", "i", "o"]} {
    %x = tessera.gather %table, %tokens : (tensor<7x5xf32>, tensor<1xi32>) -> tensor<1x5xf32>
    return %x : tensor<1x5xf32>
  }
}

// SCHED-LABEL: func.func @default_axis
// SCHED: schedule.embedding_gather
// SCHED-SAME: shape = array<i64: 7, 5, 1>

// -----

// Another target's module is not this contract's: x86 and Apple keep their own owners.
module attributes {tessera.target = "cpu"} {
  func.func @untouched(%table: tensor<8x4xf32>, %tokens: tensor<2xi32>) -> tensor<2x4xf32> {
    %x = tessera.gather %table, %tokens {axis = 0 : i64} : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
    return %x : tensor<2x4xf32>
  }
}

// SCHED-LABEL: func.func @untouched
// SCHED-NOT: schedule.embedding_gather
// SCHED: tessera.gather
