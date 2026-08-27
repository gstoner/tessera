// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s

// The shared layout authority proves a static affine i64 base.  ROCm consumes
// that base through the pre-existing tile.linear_base view carrier; it does
// not reinterpret nested/dynamic layout structure in this physical pass.
#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>
#mem = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>

func.func @gfx1151_composed_layout_linear_base(%base: !llvm.ptr, %r: i64, %c: i64) -> !tile.tile {
  %linear = "tile.materialize_composed_layout"(%r, %c) {layout = #tile.composed_layout<[16, 16], [16, 1], [[[16], [1]], [[16], [1]]], [0, 0]>} : (i64, i64) -> i64
  %view = tile.view %base, %linear, %r, %c {tile.linear_base, tile.layout = #lay, tile.memory = #mem} : (!llvm.ptr, i64, i64, i64) -> !tile.tile
  return %view : !tile.tile
}

func.func @gfx1151_dynamic_nested_composed_layout_linear_base(
    %base: !llvm.ptr, %r: i64, %c: i64, %m: i64, %lda: i64) -> !tile.tile {
  %linear = "tile.materialize_composed_layout"(%r, %c, %m, %lda) {layout = #tile.composed_layout<[[-1], [16]], [[-1], [1]], [[[16], [1]], [[16], [1]]], [0, 0]>} : (i64, i64, i64, i64) -> i64
  %view = tile.view %base, %linear, %r, %c, %m, %m, %lda {tile.linear_base, tile.layout = #lay, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (!llvm.ptr, i64, i64, i64, i64, i64, i64) -> !tile.tile
  return %view : !tile.tile
}

func.func @gfx1151_tuple_basis_composed_layout_linear_base(%c: i64) -> i64 {
  %linear = "tile.materialize_composed_layout"(%c) {layout = #tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [0]>} : (i64) -> i64
  return %linear : i64
}

func.func @gfx1151_tuple_codomain_composed_layout_linear_base(%c: i64) -> (i64, i64) {
  %pair:2 = "tile.materialize_composed_layout_tuple"(%c) {layouts = [#tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [0]>, #tile.composed_layout<[8], [2], [[[4, 2], [1, 4]]], [3]>]} : (i64) -> (i64, i64)
  return %pair#0, %pair#1 : i64, i64
}

// CHECK-LABEL: func.func @gfx1151_composed_layout_linear_base
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.divui
// CHECK: arith.muli {{.*}}, %c16_i64
// CHECK: tile.view %arg0, {{.*}}, %arg1, %arg2
// CHECK-NOT: tile.materialize_composed_layout
// CHECK-LABEL: func.func @gfx1151_dynamic_nested_composed_layout_linear_base
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.divui
// CHECK: arith.muli {{.*}}, %arg4
// CHECK: tile.view %arg0, {{.*}}, %arg1, %arg2, %arg3, %arg3, %arg4
// CHECK-NOT: tile.materialize_composed_layout
// CHECK-LABEL: func.func @gfx1151_tuple_basis_composed_layout_linear_base
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK-NOT: tile.materialize_composed_layout
// CHECK-LABEL: func.func @gfx1151_tuple_codomain_composed_layout_linear_base
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK-NOT: tile.materialize_composed_layout
