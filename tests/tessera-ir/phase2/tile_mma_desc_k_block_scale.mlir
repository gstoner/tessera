// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s

// ROCM-MACRO-K-TILE-1 / ROCM-FP8-BLOCKSCALE-1. `k_blocks` is the macro K tile;
// `scale_k` is the semantic contraction extent sharing one scale. A scale group
// must contain whole instruction-K steps and the macro tile must contain whole
// scale groups. The two extents are intentionally not equal: MXFP4 fixes K=32
// while the schedule can independently select macro K=64 or 128.
//
// Both negative cases below are numerically silent without this check: the GEMM
// runs, and the scale lands on the wrong span of the contraction.

// A scale group that cuts through an instruction cannot be applied exactly.
// CHECK: scale_k (24) must be a multiple of instruction K (16)
#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "e4m3", b = "e4m3", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 4, scale_k = 24, scale_fmt = "fp32">
func.func @scale_group_cuts_instruction() attributes {mma = #mma} { return }

// -----

// A macro tile that cuts through a scale group cannot close its accumulator.
// CHECK: macro K block, k * k_blocks (16 * 3 = 48), must be a multiple of scale_k (32)
#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "e4m3", b = "e4m3", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 3, scale_k = 32, scale_fmt = "fp32">
func.func @macro_block_cuts_scale_group() attributes {mma = #mma} { return }

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
