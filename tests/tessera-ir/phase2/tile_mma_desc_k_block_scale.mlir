// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s

// ROCM-MACRO-K-TILE-1 / ROCM-FP8-BLOCKSCALE-1. `k_blocks` IS the macro K tile:
// the descriptor could always state it, and every consumer refused anything but
// 1, which is what made the K tile unreachable. `scale_k` is the contraction
// extent sharing one scale, and the two are bound together -- the scale group
// must be EXACTLY one K block, which is the invariant AITER's reference kernel
// asserts twice as `GROUP_K == BLOCK_SIZE_K`.
//
// Both negative cases below are numerically silent without this check: the GEMM
// runs, and the scale lands on the wrong span of the contraction.

// A scale group larger than the K block spans two accumulate boundaries.
// CHECK: scale_k (128) must equal the K block, k * k_blocks (16 * 1 = 16)
#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "e4m3", b = "e4m3", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1, scale_k = 128, scale_fmt = "fp32">
func.func @scale_group_wider_than_block() attributes {mma = #mma} { return }

// -----

// A scale group smaller than the K block scales a partial product.
// CHECK: scale_k (64) must equal the K block, k * k_blocks (16 * 8 = 128)
#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "e4m3", b = "e4m3", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 8, scale_k = 64, scale_fmt = "fp32">
func.func @scale_group_narrower_than_block() attributes {mma = #mma} { return }

// -----

// A format with no block carries nothing.
// CHECK: scale_fmt "e8m0" is set but scale_k is 0
#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "e4m3", b = "e4m3", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 8, scale_k = 0, scale_fmt = "e8m0">
func.func @format_without_block() attributes {mma = #mma} { return }

// -----

// A block with no format leaves the element form to a default, which a
// numeric contract may not do (Decision #21a).
// CHECK: scale_k is 128 but scale_fmt is empty
#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "e4m3", b = "e4m3", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 8, scale_k = 128, scale_fmt = "">
func.func @block_without_format() attributes {mma = #mma} { return }
