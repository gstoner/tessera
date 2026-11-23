// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE
//
// AR-DECODE-PROGRAM-1 S1b: an sm_120 row argmax/argmin descends Graph ->
// Schedule -> Tile with a content-addressed native contract that names the
// tie-break (first occurrence), the NaN rule (first NaN wins) and the
// physical schedule (one 128-thread CTA per row, warp-shuffle combine).

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @logits(%x: tensor<3x256xf32>) -> tensor<3xi32>
      attributes {tessera.bindings = ["logits", "token"]} {
    %i = tessera.argmax %x {axis = -1 : i64, keepdims = false} : (tensor<3x256xf32>) -> tensor<3xi32>
    return %i : tensor<3xi32>
  }
}

// SCHED-LABEL: func.func @logits
// SCHED: tessera.argmax {{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHED: schedule.argreduce
// SCHED-SAME: artifact_hash = "[[HASH]]"
// SCHED-SAME: axis_mode = "last"
// SCHED-SAME: kind = "argmax"
// SCHED-SAME: nan_mode = "first_nan"
// SCHED-SAME: schedule = "cta_128_warp_shuffle"
// SCHED-SAME: shape = array<i64: 3, 256>
// SCHED-SAME: tie_break = "first"

// TILE: llvm.func @tessera_tile_argreduce_argmax_f32_{{[0-9a-f]+}}(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64)
// TILE-SAME: nvvm.kernel
// TILE-SAME: nvvm.reqntid = array<i32: 128, 1, 1>
// TILE: tile.argreduce_kernel
// TILE-SAME: kind = "argmax", nan_mode = "first_nan", output_storage = "i32", schedule = "cta_128_warp_shuffle", storage = "f32", tie_break = "first"
// TILE-NOT: func.func @logits

// -----

// No axis flattens the operand (NumPy axis=None); keepdims keeps rank.
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @flat(%x: tensor<2x3x5xf32>) -> tensor<1x1x1xi32>
      attributes {tessera.bindings = ["x", "idx"]} {
    %i = tessera.argmin %x {keepdims = true} : (tensor<2x3x5xf32>) -> tensor<1x1x1xi32>
    return %i : tensor<1x1x1xi32>
  }
}

// SCHED-LABEL: func.func @flat
// SCHED: schedule.argreduce
// SCHED-SAME: axis_mode = "flatten"
// SCHED-SAME: keepdims = true
// SCHED-SAME: kind = "argmin"
// SCHED-SAME: logical_shape = array<i64: 2, 3, 5>
// SCHED-SAME: result_shape = array<i64: 1, 1, 1>
// SCHED-SAME: shape = array<i64: 1, 30>

// TILE: llvm.func @tessera_tile_argreduce_argmin_f32_
// TILE: tile.argreduce_kernel
// TILE-SAME: kind = "argmin"

// -----

// Another target's module is not this contract's: x86 keeps its own owner.
module attributes {tessera.target = "cpu"} {
  func.func @untouched(%x: tensor<3x17xf32>) -> tensor<3xi32> {
    %i = tessera.argmax %x {axis = -1 : i64} : (tensor<3x17xf32>) -> tensor<3xi32>
    return %i : tensor<3xi32>
  }
}

// SCHED-LABEL: func.func @untouched
// SCHED-NOT: schedule.argreduce
// SCHED: tessera.argmax
