// RUN: not %tnv --lower-tile-to-nvidia='sm=100' %s 2>&1 | FileCheck %s

// TILE-LATENT-DEFECTS-2026-09-27. The `!tile.tmem` handle lowers to the i32
// tensor-memory address only when every consumer of it is lowered by this
// pass. `tile.tcgen05.mma` has no NVIDIA lowering today, so a handle it
// consumes cannot be rewritten; the pass refuses instead of erasing the
// allocation under a live use (Decision #21).

module {
  func.func @tcgen05_consumer(
      %lhs: !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "tcgen05">,
      %rhs: !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "tcgen05">) {
    %acc = tile.tmem.allocate {bytes = 16384 : i64, alignment = 128 : i64}
        : !tile.tmem
    %next = tile.tcgen05.mma %lhs, %rhs, %acc {
      cta_group = 2 : i64,
      mma = #tile.mma_desc<family = "tcgen05", m = 64, n = 64, k = 32,
        a = "bf16", b = "bf16", acc = "f32",
        a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "tcgen05">,
        !tile.fragment<m = 64, n = 64, k = 32, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "tcgen05">,
        !tile.tmem -> !tile.tmem
    return
  }
}

// CHECK: NVIDIA_TMEM_HANDLE_UNLOWERED: 'tile.tcgen05.mma' consumes a TMEM handle and has no NVIDIA lowering onto a TMEM address (NVIDIA sm_100)
