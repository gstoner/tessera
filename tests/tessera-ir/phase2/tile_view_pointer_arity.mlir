// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
//
// The pointer-backed `tile.view` operand contract, checked in the SHARED
// verifier rather than per backend (PR #510 review).
//
//   3 inputs: (base, rowOrigin, colOrigin)
//   5 inputs: + (rowBound, colBound) — the logical extents, runtime values on a
//             ragged problem, so a fragment straddling the edge can be masked.
//
// `ViewOp::verify` used to accept any count >= 3. That made a 4-operand view
// legal and meaningless, and left "is the bounded form valid?" to whichever
// backend happened to look — ROCm's materializer accepted 3, NVIDIA's still
// requires exactly 3, so the same valid Tile IR compiled on one target and
// failed on the other. A shared-IR op whose accepted shape is decided by its
// consumers is not a contract.

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#mem = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>

func.func @four_operands_is_neither_form(%b: memref<?xf16>, %r: index,
                                         %c: index, %x: index) {
  // CHECK: TILE_VIEW_POINTER_ARITY
  // CHECK-SAME: got 4 operands
  %v = tile.view %b, %r, %c, %x {tile.layout = #lay, tile.memory = #mem}
      : (memref<?xf16>, index, index, index) -> !tile.tile
  return
}

// -----

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#mem = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>

// Six is not "five plus something harmless" — an extra operand means the
// producer believes it is passing information the contract does not define.
func.func @six_operands_is_rejected(%b: memref<?xf16>, %r: index, %c: index,
                                    %rb: index, %cb: index, %extra: index) {
  // CHECK: TILE_VIEW_POINTER_ARITY
  // CHECK-SAME: got 6 operands
  %v = tile.view %b, %r, %c, %rb, %cb, %extra
      {tile.layout = #lay, tile.memory = #mem}
      : (memref<?xf16>, index, index, index, index, index) -> !tile.tile
  return
}
