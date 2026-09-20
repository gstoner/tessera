// RUN: tessera-opt %s | FileCheck %s

// The positive side: a macro K tile of 128 over a 16-wide instruction is
// k_blocks = 8, with the scale group exactly matching it. `k_blocks > 1` used
// to be refused by every consumer (ROCM-MACRO-K-TILE-1).
// CHECK: k_blocks = 8
// CHECK-SAME: scale_k = 128
// CHECK-SAME: scale_fmt = "fp32"
#scaled = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                         a = "e4m3", b = "e4m3", acc = "f32",
                         a_layout = "row_major", b_layout = "col_major",
                         k_blocks = 8, scale_k = 128, scale_fmt = "fp32">
func.func @block_scaled() attributes {mma = #scaled} { return }

// An unscaled descriptor still prints without the scale clause at all, which is
// what keeps the 45 fixtures that print an mma_desc unchanged.
// CHECK: k_blocks = 8>
// CHECK-NOT: scale_k
#unscaled = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                           a = "f16", b = "f16", acc = "f32",
                           a_layout = "row_major", b_layout = "col_major",
                           k_blocks = 8>
func.func @unscaled_k_block() attributes {mma = #unscaled} { return }
