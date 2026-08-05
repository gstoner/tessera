// RUN: tessera-opt %s --split-input-file --lower-tile-to-rocm=arch=gfx1151 \
// RUN:   | FileCheck %s
//
// W1.1 step 3 prerequisite — a fragment whose K axis is STRIDED in memory.
//
// `materializeFragmentPack` addressed only fragments whose K axis is contiguous,
// and rejected the rest as an "unsupported source layout". The fragment always
// walks K; the memory order decides whether that walk is contiguous:
//
//   role  order        K axis   linear             K stride
//   a     row_major    col      row * ld + col     1
//   a     col_major    col      col * ld + row     ld
//   b     row_major    row      row * ld + col     ld
//   b     col_major    row      col * ld + row     1
//
// Only rows 1 and 4 were reachable. That is the concrete reason
// `GenerateWMMAGemmKernel` could not migrate onto the typed form: it stores B
// row-major (`k * N + col`), so its B fragment is a stride-N gather — which is
// why the generator assembles B by hand with 16 scalar loads and inserts, the
// very lane math the typed contract is supposed to own.
//
// A strided fragment cannot use `vector.load` or `vector.maskedload` (both want
// contiguity), so it gathers element by element with the same
// `inb ? value : zero` shape the generator uses. That equivalence is what will
// let the migrated producer stay bit-identical rather than merely close.

#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "f16", b = "f16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
#layA = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                     replica = [] : [] on [], offset = 0>
#layB = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                     replica = [] : [] on [], offset = 0>
#memRow = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>

// B stored ROW-major (K x N, ld = N) — the production generator's layout.
// The gather multiplies the element ordinal by the leading dimension, and there
// is no vector load at all on this path.
//
// A is row-major and therefore CONTIGUOUS in K — it keeps its single
// `vector.load`. Only B changes. Asserting "no vector.load" here would be
// wrong, and an earlier version of this file did exactly that: it passed
// vacuously, because the directive was scoped to the lines before A's load.
//
// CHECK-LABEL: func.func @row_major_b_gathers_at_stride_ld
// CHECK: vector.load
// B is strided: one scalar load per K element, 16 of them, each inserted at its
// own ordinal. `vector.maskedload` cannot appear at all — it needs contiguity.
// CHECK-NOT: vector.maskedload
// CHECK-COUNT-16: memref.load
// The last ordinal proves the full K walk rather than a truncated one.
// CHECK: vector.insert %{{.*}}, %{{.*}} [15]
// CHECK: tessera_rocm.wmma
func.func @row_major_b_gathers_at_stride_ld(
    %a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf32>,
    %r: index, %c: index) {
  %va = tile.view %a, %r, %c {tile.layout = #layA, tile.memory = #memRow}
      : (memref<?xf16>, index, index) -> !tile.tile
  %vb = tile.view %b, %r, %c {tile.layout = #layB, tile.memory = #memRow}
      : (memref<?xf16>, index, index) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %vb {role = "b", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment
  %res = tile.mma %fa, %fb, %acc {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  %o = tile.fragment_unpack %res {role = "acc", mma = #mma, tile.layout = #layA}
      : (!tile.fragment) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #layA, tile.memory = #memRow}
      : !tile.tile, memref<?xf32>, index, index
  return
}

// -----

#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "f16", b = "f16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
#layA = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                     replica = [] : [] on [], offset = 0>
#layB = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                     replica = [] : [] on [], offset = 0>
#memRow = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>

// The RAGGED strided case. Every element gets its own two-axis guard, and the
// ADDRESS is clamped as well as the value: selecting on the loaded value alone
// still performs the load, so an out-of-bounds lane would read past the matrix
// before the select could discard it.
//
// CHECK-LABEL: func.func @ragged_row_major_b_guards_every_element
// A stays contiguous, so it keeps the mask + maskedload form.
// CHECK: vector.maskedload
// One element's full shape: two-axis guard, clamped ADDRESS, load, clamped
// VALUE, insert. These ops interleave per element, so this is checked once as a
// sequence rather than as two separate CHECK-COUNT blocks — those cannot both
// match interleaved text, which is how an earlier version of this file failed.
// CHECK: arith.andi
// CHECK: arith.select
// CHECK: memref.load
// CHECK: arith.select
// CHECK: vector.insert
// Then the remaining 15 elements, ending at the last ordinal — which is what
// proves the full K walk rather than a truncated one.
// CHECK-COUNT-15: memref.load
// CHECK: vector.insert %{{.*}}, %{{.*}} [15]
func.func @ragged_row_major_b_guards_every_element(
    %a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf32>,
    %r: index, %c: index, %rb: index, %cb: index) {
  %va = tile.view %a, %r, %c, %rb, %cb
      {tile.layout = #layA, tile.memory = #memRow}
      : (memref<?xf16>, index, index, index, index) -> !tile.tile
  %vb = tile.view %b, %r, %c, %rb, %cb
      {tile.layout = #layB, tile.memory = #memRow}
      : (memref<?xf16>, index, index, index, index) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %vb {role = "b", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment
  %res = tile.mma %fa, %fb, %acc {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  %o = tile.fragment_unpack %res {role = "acc", mma = #mma, tile.layout = #layA}
      : (!tile.fragment) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #layA, tile.memory = #memRow}
      : !tile.tile, memref<?xf32>, index, index
  return
}
