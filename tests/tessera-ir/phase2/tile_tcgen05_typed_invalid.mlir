// RUN: tessera-opt %s -verify-diagnostics

// CAKE §5.2 row 8: TCGen05 consumes the same parameterized fragment ABI as
// tile.mma. The historical bare migration type may parse, but cannot execute.
func.func @bare_tcgen05_fragment_is_rejected(
    %lhs: !tile.fragment, %rhs: !tile.fragment, %acc: !tile.tmem) {
  // expected-error@+1 {{TILE_TCGEN05_FRAGMENT_CONTRACT}}
  %next = tile.tcgen05.mma %lhs, %rhs, %acc {
    cta_group = 2 : i64,
    mma = #tile.mma_desc<family = "tcgen05", m = 64, n = 64, k = 32,
      a = "bf16", b = "bf16", acc = "f32",
      a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
  } : !tile.fragment, !tile.fragment, !tile.tmem -> !tile.tmem
  return
}
