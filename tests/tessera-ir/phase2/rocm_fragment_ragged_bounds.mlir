// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --split-input-file --lower-tile-to-rocm=arch=gfx1151 \
// RUN:   | FileCheck %s
//
// W1.1 step 3 — ragged-edge masking for typed fragments.
//
// `materializeFragmentPack` loaded `inputElementsPerLane` contiguous elements
// with an unguarded `vector.load`, so a fragment straddling the edge of a
// ragged matrix read past it. `GenerateWMMAGemmKernel` avoids that by
// assembling fragments element by element with an `inb ? value : zero` select —
// lane math the typed contract is supposed to own, and precisely why that
// producer could not migrate to the typed form (design doc §4.5).
//
// `tile.view` is pointer-backed as (base, rowOrigin, colOrigin). It may now
// carry two more operands, (rowBound, colBound): the logical extents, which are
// runtime values on a ragged problem. The guard previously required EXACTLY 3
// inputs, so the bounded form was unreachable — and silently, because it fell
// into the generic "unsupported source layout" diagnostic rather than naming
// the arity.
//
// Bounded fragments use explicit guarded scalar loads. This reproduces the
// generator's `inb ? value : zero` sequence and stays legal through ROCm's
// GPU-to-LLVM pipeline, which does not lower vector.create_mask/maskedload.
//
// The fast axis differs by role: A is row-major so the load walks columns; B is
// col-major (`col * leadingDim + row`) so it walks rows. Masking the wrong axis
// would silently zero live data on exactly the ragged shapes this protects.

#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "f16", b = "f16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>
#memB = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 64>

// CHECK-LABEL: func.func @ragged_view_masks_every_load
// CHECK-NOT: vector.maskedload
// CHECK-NOT: vector.create_mask
// CHECK-COUNT-32: memref.load
// CHECK: vector.insert {{.*}} [15]
func.func @ragged_view_masks_every_load(
    %base: memref<?xf16>, %out: memref<?xf32>,
    %r: index, %c: index, %rb: index, %cb: index) {
  %va = tile.view %base, %r, %c, %rb, %cb
      {tile.layout = #lay, tile.memory = #memA}
      : (memref<?xf16>, index, index, index, index) -> !tile.tile
  %vb = tile.view %base, %r, %c, %rb, %cb
      {tile.layout = #lay, tile.memory = #memB}
      : (memref<?xf16>, index, index, index, index) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %vb {role = "b", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment
  %res = tile.mma %fa, %fb, %acc {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  %o = tile.fragment_unpack %res {role = "acc", mma = #mma, tile.layout = #lay}
      : (!tile.fragment) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : !tile.tile, memref<?xf32>, index, index
  return
}

// -----

#mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                      a = "f16", b = "f16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>
#memB = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 64>

// The 3-input form is unchanged: no bounds, no mask, plain load. Every producer
// in the tree emits this today, so a regression here is a silent codegen change
// on the working lane rather than a new feature misbehaving.
// CHECK-LABEL: func.func @unbounded_view_loads_unmasked
// CHECK-NOT: vector.create_mask
// CHECK-NOT: vector.maskedload
// CHECK: vector.load
func.func @unbounded_view_loads_unmasked(
    %base: memref<?xf16>, %out: memref<?xf32>, %r: index, %c: index) {
  %va = tile.view %base, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : (memref<?xf16>, index, index) -> !tile.tile
  %vb = tile.view %base, %r, %c {tile.layout = #lay, tile.memory = #memB}
      : (memref<?xf16>, index, index) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %vb {role = "b", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment
  %res = tile.mma %fa, %fb, %acc {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  %o = tile.fragment_unpack %res {role = "acc", mma = #mma, tile.layout = #lay}
      : (!tile.fragment) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : !tile.tile, memref<?xf32>, index, index
  return
}
