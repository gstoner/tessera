// RUN: tessera-opt --tessera-tile-to-x86='architecture=base prefer-amx=false' %s | FileCheck %s

module {
  func.func @static_mixed_radix(%coordinate: i64) -> i64 {
    %linear = "tile.materialize_composed_layout"(%coordinate) {
      layout = #tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [3]>
    } : (i64) -> i64
    return %linear : i64
  }

  func.func @bounded_dynamic(%row: i64, %column: i64, %m: i64,
                             %lda: i64) -> i64 {
    %linear = "tile.materialize_composed_layout"(%row, %column, %m, %lda) {
      layout = #tile.composed_layout<[[-1], [16]], [[-1], [1]], [[[16], [1]], [[16], [1]]], [0, 0]>
    } : (i64, i64, i64, i64) -> i64
    return %linear : i64
  }

  func.func @tuple_product(%coordinate: i64) -> (i64, i64) {
    %pair:2 = "tile.materialize_composed_layout_tuple"(%coordinate) {
      layouts = [
        #tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [0]>,
        #tile.composed_layout<[8], [2], [[[4, 2], [1, 4]]], [3]>
      ]
    } : (i64) -> (i64, i64)
    return %pair#0, %pair#1 : i64, i64
  }
}

// CHECK-LABEL: func.func @static_mixed_radix
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate must be nonnegative"
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate exceeds its outer extent"
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK-NOT: tile.materialize_composed_layout

// CHECK-LABEL: func.func @bounded_dynamic
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate must be nonnegative"
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate must be nonnegative"
// CHECK: cf.assert {{.*}}, "x86 composed-layout dynamic extent must be positive"
// CHECK: cf.assert {{.*}}, "x86 composed-layout dynamic stride must be nonnegative"
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate exceeds its outer extent"
// CHECK: arith.muli {{.*}}, %arg3
// CHECK-NOT: tile.materialize_composed_layout

// CHECK-LABEL: func.func @tuple_product
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate must be nonnegative"
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate exceeds its outer extent"
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate must be nonnegative"
// CHECK: cf.assert {{.*}}, "x86 composed-layout coordinate exceeds its outer extent"
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK-NOT: tile.materialize_composed_layout
