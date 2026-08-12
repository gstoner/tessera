// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --lower-tile-to-rocm=arch=gfx1200 | FileCheck %s
//
// W1.1 step 0 — the accumulator's input dtype is SEARCHED, not assumed.
//
// An `acc` fragment names no input dtype, so resolving its physical layout
// needs one supplied. The first implementation pinned a single representative
// ("int8" for i32, "f16" for f32). That is wrong wherever
// `resolveFragmentLayout` derives the legal `k` FROM the dtype — which it does
// on every architecture:
//
//   RDNA4   int4 -> k=32   but int8 -> k=16     <-- this file
//   gfx125x fp8  -> k=64   but f16  -> k=32
//   CDNA    f32  -> k=8, f16 -> k=16, fp8 -> k=32, fp4 -> k=64
//
// So a supported int4 MMA had a *resolvable* A/B pair and an *unresolvable*
// accumulator, and failed to lower with ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR
// naming the accumulator — a confusing report of a legal program.
//
// Searching is sound because the accumulator width does not depend on which
// candidate matches: every branch sets both accumulator fields to
// `256 / waveSize`, and `waveSize` is fixed per architecture. The search only
// has to find one candidate, not the right one. Candidates preserve
// integer-ness, which is what selects the i32-vs-f32 accumulator element type.
//
// gfx1200 is RDNA 4: `wmma_i32_16x16x32_iu4`. int4 rides in i8 storage.

// Shard extents are per ROLE: A is {M,K}, B is {K,N}, the accumulator {M,N}.
#layA = #tile.layout<shard = [16, 32] : [32, 1] on ["tlane", "reg"],
                     replica = [] : [] on [], offset = 0>
#layB = #tile.layout<shard = [32, 16] : [16, 1] on ["tlane", "reg"],
                     replica = [] : [] on [], offset = 0>
#layC = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                     replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>
#memB = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 64>

!fa  = !tile.fragment<m = 16, n = 16, k = 32, elem = "int4", acc = "i32",
                      role = "a", layout = "row_major", family = "wmma">
!fb  = !tile.fragment<m = 16, n = 16, k = 32, elem = "int4", acc = "i32",
                      role = "b", layout = "col_major", family = "wmma">
!fac = !tile.fragment<m = 16, n = 16, k = 32, elem = "i32", acc = "i32",
                      role = "acc", layout = "row_major", family = "wmma">

// CHECK-LABEL: func.func @rdna4_int4_accumulator_resolves
// The accumulator is 256/32 = 8 lanes of i32, independent of the input dtype.
// CHECK: tessera_rocm.wmma
// CHECK-SAME: accum = "i32"
// CHECK-SAME: input_dtype = "int4"
// CHECK-SAME: shape = "m16n16k32"
// CHECK-SAME: -> vector<8xi32>
func.func @rdna4_int4_accumulator_resolves(
    %a: memref<?xi8>, %b: memref<?xi8>, %out: memref<?xi32>,
    %r: index, %c: index) {
  %acc = tile.fragment_zero : !fac
  %va = tile.view %a, %r, %c {tile.layout = #layA, tile.memory = #memA}
      : (memref<?xi8>, index, index) -> !tile.tile
  %vb = tile.view %b, %r, %c {tile.layout = #layB, tile.memory = #memB}
      : (memref<?xi8>, index, index) -> !tile.tile
  %fa = tile.fragment_pack %va : (!tile.tile) -> !fa
  %fb = tile.fragment_pack %vb : (!tile.tile) -> !fb
  %m = tile.mma %fa, %fb, %acc : (!fa, !fb, !fac) -> !fac
  %o = tile.fragment_unpack %m {tile.layout = #layC} : (!fac) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #layC, tile.memory = #memA}
      : !tile.tile, memref<?xi32>, index, index
  return
}
