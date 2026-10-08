// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --implicit-check-not=f64

// E8M0 code 0 has exponent -127, not a zero value. Native ldexp composes
// exponents before scaling, matching the wide reference with one f32 rounding.
// CHECK-LABEL: func.func @e8m0_join
// CHECK: memref.load
// CHECK: arith.extui
// CHECK: arith.constant 254 : i32
// CHECK: arith.constant 255 : i32
// CHECK: arith.addi {{.*}} : i32
// CHECK: arith.subi {{.*}} : i32
// CHECK: llvm.intr.ldexp({{.*}}) : (f32, i32) -> f32
// CHECK: arith.constant 2143289344 : i32
// CHECK: arith.bitcast
// CHECK: arith.ori
// CHECK: arith.select
// CHECK: arith.addf {{.*}} : f32
// CHECK-NOT: tile.fragment_scaled_accumulate
// CHECK: return

!acc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
func.func @e8m0_join(%sa: memref<?xi8>, %sb: memref<?xi8>,
                    %row: index, %col: index, %group: index, %groups: index,
                    %rows: index, %cols: index) {
  %acc = tile.fragment_zero : !acc
  %partial = tile.fragment_zero : !acc
  %out = tile.fragment_scaled_accumulate %acc, %partial scales(%sa, %sb) at(%row, %col)
      group(%group, %groups) bounds(%rows, %cols)
      {scale_n = 1 : i64, scale_format = "e8m0"} : !acc, memref<?xi8>, memref<?xi8>
  return
}
