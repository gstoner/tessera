// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s

// ROCM-FP8-BLOCKSCALE-1 (FOUNDATION-BATCH-3-2026-09-28). A weight-scale block
// that is a whole number of fragment widths (scale_n % 16 == 0) holds every
// column of a fragment whose origin is PROVABLY 16-aligned, so the weight
// scale is loaded once per fragment. The proof is derived from the origin's
// arithmetic; an origin it cannot prove keeps one load per element, and so
// does a block narrower than a fragment.

!acc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">

// CHECK-LABEL: func.func @aligned_origin(
// CHECK-SAME: memref<?xf32>, %[[SB:[a-z0-9_]+]]: memref<?xf32>
// CHECK-COUNT-1: memref.load %[[SB]][
// CHECK-NOT: memref.load %[[SB]][
// CHECK: return
func.func @aligned_origin(%sa: memref<?xf32>, %sb: memref<?xf32>, %blk: index,
                          %row: index, %g: index, %gs: index, %m: index, %n: index) {
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %blk, %c64 : index
  %col = arith.addi %base, %c16 : index
  %acc = tile.fragment_zero : !acc
  %part = tile.fragment_zero : !acc
  %0 = tile.fragment_scaled_accumulate %acc, %part scales(%sa, %sb) at(%row, %col) group(%g, %gs) bounds(%m, %n) {scale_n = 128 : i64} : !acc, memref<?xf32>, memref<?xf32>
  return
}

// CHECK-LABEL: func.func @unproven_origin(
// CHECK-SAME: memref<?xf32>, %[[SB:[a-z0-9_]+]]: memref<?xf32>
// CHECK-COUNT-8: memref.load %[[SB]][
// CHECK-NOT: memref.load %[[SB]][
// CHECK: return
func.func @unproven_origin(%sa: memref<?xf32>, %sb: memref<?xf32>, %col: index,
                           %row: index, %g: index, %gs: index, %m: index, %n: index) {
  %acc = tile.fragment_zero : !acc
  %part = tile.fragment_zero : !acc
  %0 = tile.fragment_scaled_accumulate %acc, %part scales(%sa, %sb) at(%row, %col) group(%g, %gs) bounds(%m, %n) {scale_n = 128 : i64} : !acc, memref<?xf32>, memref<?xf32>
  return
}

// CHECK-LABEL: func.func @narrow_block(
// CHECK-SAME: memref<?xf32>, %[[SB:[a-z0-9_]+]]: memref<?xf32>
// CHECK-COUNT-8: memref.load %[[SB]][
// CHECK-NOT: memref.load %[[SB]][
// CHECK: return
func.func @narrow_block(%sa: memref<?xf32>, %sb: memref<?xf32>, %blk: index,
                        %row: index, %g: index, %gs: index, %m: index, %n: index) {
  %c64 = arith.constant 64 : index
  %col = arith.muli %blk, %c64 : index
  %acc = tile.fragment_zero : !acc
  %part = tile.fragment_zero : !acc
  %0 = tile.fragment_scaled_accumulate %acc, %part scales(%sa, %sb) at(%row, %col) group(%g, %gs) bounds(%m, %n) {scale_n = 8 : i64} : !acc, memref<?xf32>, memref<?xf32>
  return
}
