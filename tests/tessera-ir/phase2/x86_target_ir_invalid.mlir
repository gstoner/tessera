// RUN: not tessera-opt %s 2>&1 | FileCheck %s
//
// W0.10 negative fixture — the typed x86 Target IR must REJECT, not just
// accept.
//
// This is the whole argument for building the dialect rather than granting
// Decision #19 a carve-out. The previous x86 "Target IR" was 21 `func.call`s
// into a C shim plus, on the Python side, text asserted with substring
// matches. Neither can catch an AMX dot-product whose operands never came from
// a tile: `"tessera_x86.amx_dpbf16ps" in text` is true either way.
//
// A typed dialect makes it a parse error.

func.func @dpbf16ps_rejects_non_tile_operands(%v: vector<16xf32>) {
  // CHECK: error
  // CHECK-SAME: use of value '%v' expects different type
  %res = tessera_x86.amx_dpbf16ps %v, %v, %v
      : !tessera_x86.tile, !tessera_x86.tile, !tessera_x86.tile
      -> !tessera_x86.tile
  return
}
