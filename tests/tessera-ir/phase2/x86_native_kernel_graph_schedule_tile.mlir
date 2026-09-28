// E2E-REAL-6 x86 elementwise / cohort-2 / breadth (2026-09-28): one isolated
// static Graph op -> content-addressed Schedule record -> the Tile launch
// envelope the stable x86 C ABI consumes (NativeX86Kernel.h), plus the x86
// row normalization on schedule.norm.
// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s
// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHED

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["c", "a", "b", "o"]} {
  func.func @select(%c: tensor<3x17xi1>, %a: tensor<3x17xf32>, %b: tensor<3x17xf32>) -> tensor<3x17xf32> {
    %o = tessera.where %c, %a, %b {tessera.effect_kind = "pure"} : (tensor<3x17xi1>, tensor<3x17xf32>, tensor<3x17xf32>) -> tensor<3x17xf32>
    return %o : tensor<3x17xf32>
  }
}
// SCHED-LABEL: func.func @select
// SCHED: schedule.artifact {
// SCHED-SAME: shape_key = "family=x86_kernel;kind=elementwise/where"
// CHECK-LABEL: llvm.func @tessera_tile_x86_where_where
// CHECK-SAME: tessera.x86_kernel_contract = {bindings = ["c", "a", "b", "o"]
// CHECK-SAME: scalars = {N = 51 : i64}
// CHECK-SAME: storage = ["i8", "f32", "f32", "f32"]
// CHECK: tile.elementwise_kernel
// CHECK-SAME: condition_storage = "i8", family = "where", kind = "where"
// CHECK-NOT: tessera.where %

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["x", "idx"]} {
  func.func @argmax_flat(%x: tensor<3x17xf32>) -> tensor<i32> {
    %o = tessera.argmax %x {keepdims = false} : (tensor<3x17xf32>) -> tensor<i32>
    return %o : tensor<i32>
  }
}
// CHECK-LABEL: llvm.func @tessera_tile_x86_argreduce_argmax
// CHECK-SAME: axis_mode = "flatten"
// CHECK-SAME: logical_shape = array<i64: 51>
// CHECK-SAME: scalars = {Cols = 51 : i64, Rows = 1 : i64}
// CHECK: tile.argreduce_kernel
// CHECK-SAME: tie_break = "first"

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["source", "idx", "out"]} {
  func.func @gather(%s: tensor<20xf32>, %i: tensor<7xi64>) -> tensor<7xf32> {
    %o = tessera.gather %s, %i {axis = 0 : i64} : (tensor<20xf32>, tensor<7xi64>) -> tensor<7xf32>
    return %o : tensor<7xf32>
  }
}
// CHECK-LABEL: llvm.func @tessera_tile_x86_gather_f32
// CHECK-SAME: scalars = {N = 7 : i64, SourceN = 20 : i64}
// CHECK: tile.x86_abi_kernel
// CHECK-SAME: abi = "tessera.x86.gather.f32.v1"
// CHECK-SAME: symbol = "tessera_x86_gather_f32"

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["m", "r", "out"]} {
  func.func @solve(%m: tensor<4x4xf32>, %r: tensor<4x3xf32>) -> tensor<4x3xf32> {
    %o = tessera.tri_solve %m, %r {lower = false} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
    return %o : tensor<4x3xf32>
  }
}
// CHECK-LABEL: llvm.func @tessera_tile_x86_tri_solve_f32
// CHECK-SAME: scalars = {Batch = 1 : i64, Lower = 0 : i32, M = 3 : i64, N = 4 : i64}
// CHECK: tile.x86_abi_kernel

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["x", "o"]} {
  func.func @norm(%x: tensor<3x17xf32>) -> tensor<3x17xf32> {
    %o = tessera.rmsnorm %x {eps = 1.000000e-03 : f64} : (tensor<3x17xf32>) -> tensor<3x17xf32>
    return %o : tensor<3x17xf32>
  }
}
// SCHED-LABEL: func.func @norm
// SCHED: schedule.norm
// SCHED-SAME: arch = "zen5-avx512"
// SCHED-SAME: epsilon = 1.000000e-03 : f32
// SCHED-SAME: workgroup_size = 1 : i64
// CHECK-LABEL: func.func @norm
// CHECK: arith.constant 1.000000e-03 : f32
// CHECK: tile.norm_kernel
// CHECK-SAME: kind = "rmsnorm"

// -----

// Ownership is opt-in: without launch bindings an x86 module's ops pass
// through Graph -> Schedule untouched (Decision #10a negative case).
module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @unclaimed(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    %o = tessera.maximum %a, %b : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %o : tensor<8xf32>
  }
}
// SCHED-LABEL: func.func @unclaimed
// SCHED-NOT: schedule.artifact
// SCHED: tessera.maximum
