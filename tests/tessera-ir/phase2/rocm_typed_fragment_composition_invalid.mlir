// RUN: not tessera-opt %s --split-input-file --lower-tile-to-rocm=arch=gfx1151 \
// RUN:   2>&1 | FileCheck %s
//
// W1.1 step 0, negative half — a stated fragment contract this target cannot
// realize must FAIL, by name (Decision #21).
//
// This file exists because the first implementation passed the positive tests
// while being silently wrong here. A blanket identity type conversion was
// registered alongside the fragment conversion, and MLIR's `convertType` reads
// a `std::nullopt` callback result as "not applicable, try the next callback" —
// so an *unresolvable* fragment fell through to identity, was declared legal,
// and passed through untouched. The pass then emitted a `tessera_rocm.wmma`
// whose operands were still `!tile.fragment`, and **exited 0**.
//
// The second defect that hid behind it: an `acc` fragment names no input dtype,
// so its physical layout resolves via a representative one. On gfx1151 an e4m3
// A/B pair is unsupported while its f32 accumulator resolves happily — so
// checking only the accumulator declared the whole op fine. Convertibility of
// the accumulator does not imply convertibility of the inputs.
//
// gfx1151 is RDNA 3.5: WMMA F16/BF16/IU8/IU4 and NO FP8/BF8 matrix form (those
// are RDNA 4). See docs/reference/isa/rdna/.

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>
#memB = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 64>

!fa  = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32",
                      role = "a", layout = "row_major", family = "wmma">
!fb  = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32",
                      role = "b", layout = "col_major", family = "wmma">
!fac = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32",
                      role = "acc", layout = "row_major", family = "wmma">

// CHECK: ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR
func.func @fp8_inputs_have_no_gfx1151_wmma(
    %a: memref<?xf8E4M3FN>, %b: memref<?xf8E4M3FN>, %out: memref<?xf32>,
    %r: index, %c: index) {
  %acc = tile.fragment_zero {role = "acc"} : !fac
  %va = tile.view %a, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : (memref<?xf8E4M3FN>, index, index) -> !tile.tile
  %vb = tile.view %b, %r, %c {tile.layout = #lay, tile.memory = #memB}
      : (memref<?xf8E4M3FN>, index, index) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a"} : (!tile.tile) -> !fa
  %fb = tile.fragment_pack %vb {role = "b"} : (!tile.tile) -> !fb
  %m = tile.mma %fa, %fb, %acc : (!fa, !fb, !fac) -> !fac
  %o = tile.fragment_unpack %m {role = "acc", tile.layout = #lay}
      : (!fac) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : !tile.tile, memref<?xf32>, index, index
  return
}

// -----

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>

!fac = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32",
                      role = "acc", layout = "row_major", family = "wmma">

// A typed accumulator that reaches no `tile.store` has no physical form: the
// `!tile.tile` a `fragment_unpack` produces only becomes real at the store.
// Say so by name rather than leaving a stranded op behind.
//
// CHECK: ROCM_FRAGMENT_UNPACK_UNCONSUMED
func.func @unpack_without_store(%out: memref<?xf32>) -> !tile.tile {
  %acc = tile.fragment_zero {role = "acc"} : !fac
  %o = tile.fragment_unpack %acc {role = "acc", tile.layout = #lay}
      : (!fac) -> !tile.tile
  return %o : !tile.tile
}
