// This file is fixture DATA for tests/unit/test_x86_composed_layout_exec.py,
// which drives it through tessera-tile-to-x86 -> LLVM -> lli. It is not a lit
// test: without this marker lit discovers it, reports Unresolved ("Test has no
// 'RUN:' line"), and fails `check-tessera-ir` for the whole repository.
// UNSUPPORTED: true
// Exact CPU execution fixture. The engineering gate lowers this file through
// tessera-tile-to-x86, upstream LLVM conversion, and lli; main returns zero
// only when dynamic, nested mixed-radix, and tuple-product results all match.

module {
  func.func @main() -> i32 {
    %row = arith.constant 3 : i64
    %column = arith.constant 5 : i64
    %m = arith.constant 17 : i64
    %lda = arith.constant 29 : i64
    %coordinate = arith.constant 7 : i64

    %dynamic = "tile.materialize_composed_layout"(%row, %column, %m, %lda) {
      layout = #tile.composed_layout<[[-1], [16]], [[-1], [1]], [[[16], [1]], [[16], [1]]], [0, 0]>
    } : (i64, i64, i64, i64) -> i64
    %nested = "tile.materialize_composed_layout"(%coordinate) {
      layout = #tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [3]>
    } : (i64) -> i64
    %pair:2 = "tile.materialize_composed_layout_tuple"(%coordinate) {
      layouts = [
        #tile.composed_layout<[8], [1], [[[2, 4], [1, 2]]], [0]>,
        #tile.composed_layout<[8], [2], [[[4, 2], [1, 4]]], [3]>
      ]
    } : (i64) -> (i64, i64)

    %expected_dynamic = arith.constant 92 : i64
    %expected_nested = arith.constant 10 : i64
    %expected_first = arith.constant 7 : i64
    %expected_second = arith.constant 20 : i64
    %dynamic_ok = arith.cmpi eq, %dynamic, %expected_dynamic : i64
    %nested_ok = arith.cmpi eq, %nested, %expected_nested : i64
    %first_ok = arith.cmpi eq, %pair#0, %expected_first : i64
    %second_ok = arith.cmpi eq, %pair#1, %expected_second : i64
    %left = arith.andi %dynamic_ok, %nested_ok : i1
    %right = arith.andi %first_ok, %second_ok : i1
    %all = arith.andi %left, %right : i1
    %success = arith.constant 0 : i32
    %failure = arith.constant 1 : i32
    %status = arith.select %all, %success, %failure : i32
    return %status : i32
  }
}
