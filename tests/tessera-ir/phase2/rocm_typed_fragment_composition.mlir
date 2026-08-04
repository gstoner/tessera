// RUN: tessera-opt %s --split-input-file --lower-tile-to-rocm=arch=gfx1151 \
// RUN:   | FileCheck %s
//
// W1.1 step 0 — the typed fragment path lowers by COMPOSITION.
//
// The legacy typed path is a single-shot whole-chain pattern match: from
// `tile.mma` it chases operands for `fragment_pack` A/B and a `fragment_zero`
// accumulator, requires a `fragment_unpack -> tile.store` consumer, emits one
// physical op, and erases the whole chain. Three shapes are therefore
// inexpressible BY CONSTRUCTION, and this file is one function per shape:
//
//   1. an accumulator arriving from an `scf.for` iter-arg — a K-loop, i.e.
//      what a GEMM *is*. `tile_mma_typed_kloop.mlir` proved this VERIFIES;
//      until now nothing could lower it.
//   2. a `tile.mma` whose accumulator is another `tile.mma`'s result.
//   3. an accumulator that is not a literal `fragment_zero` at all.
//
// None of these needed a new pattern. They fall out of converting
// `!tile.fragment<...>` to its physical per-lane `vector<N x T>` and letting
// each op lower against its already-converted operands. `scf.for` iter-args are
// upstream's `populateSCFStructuralTypeConversionsAndLegality`, not ours.
//
// What each function asserts is that the accumulator OPERAND of the emitted
// `tessera_rocm.wmma` is the value that flowed in. The legacy path synthesized
// a fresh zero there and discarded whatever was passed, so checking merely that
// a `wmma` was emitted would pass against the defect this closes.

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>
#memB = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 64>

!fa  = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32",
                      role = "a", layout = "row_major", family = "wmma">
!fb  = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32",
                      role = "b", layout = "col_major", family = "wmma">
!fac = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32",
                      role = "acc", layout = "row_major", family = "wmma">

// 1. The K-loop. The accumulator inside the body is a BLOCK ARGUMENT with no
// defining op, which is precisely what producer-chasing cannot see.
//
// CHECK-LABEL: func.func @k_loop_accumulator
// The loop now carries the physical accumulator, not a Tile-level type.
// CHECK: scf.for
// CHECK-SAME: iter_args(%[[ACC:.*]] = {{.*}}) -> (vector<8xf32>)
// The third operand IS the iter-arg.
// CHECK: tessera_rocm.wmma {{.*}}, {{.*}}, %[[ACC]]
// CHECK: scf.yield
func.func @k_loop_accumulator(%a: memref<?xf16>, %b: memref<?xf16>,
                              %out: memref<?xf32>, %r: index, %c: index,
                              %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc0 = tile.fragment_zero {role = "acc"} : !fac
  %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %acc0) -> (!fac) {
    %va = tile.view %a, %r, %i {tile.layout = #lay, tile.memory = #memA}
        : (memref<?xf16>, index, index) -> !tile.tile
    %vb = tile.view %b, %i, %c {tile.layout = #lay, tile.memory = #memB}
        : (memref<?xf16>, index, index) -> !tile.tile
    %fa = tile.fragment_pack %va {role = "a"} : (!tile.tile) -> !fa
    %fb = tile.fragment_pack %vb {role = "b"} : (!tile.tile) -> !fb
    %m = tile.mma %fa, %fb, %acc : (!fa, !fb, !fac) -> !fac
    scf.yield %m : !fac
  }
  %o = tile.fragment_unpack %res {role = "acc", tile.layout = #lay}
      : (!fac) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : !tile.tile, memref<?xf32>, index, index
  return
}

// -----

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>
#memB = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 64>

!fa  = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32",
                      role = "a", layout = "row_major", family = "wmma">
!fb  = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32",
                      role = "b", layout = "col_major", family = "wmma">
!fac = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32",
                      role = "acc", layout = "row_major", family = "wmma">

// 2. + 3. An unrolled two-step K reduction: the second `tile.mma` accumulates
// into the FIRST one's result. That is simultaneously an mma feeding an mma and
// an accumulator that is not a `fragment_zero` — both fatal to producer
// chasing, both unremarkable to a type conversion.
//
// CHECK-LABEL: func.func @chained_accumulator
// CHECK: %[[W0:.*]] = tessera_rocm.wmma
// The second WMMA consumes the first's RESULT as its accumulator. If the
// synthesized-zero defect regressed, this operand would be a fresh constant and
// the partial product from step 0 would be silently dropped.
// CHECK: tessera_rocm.wmma {{.*}}, {{.*}}, %[[W0]]
func.func @chained_accumulator(%a: memref<?xf16>, %b: memref<?xf16>,
                               %out: memref<?xf32>, %r: index, %c: index,
                               %k0: index, %k1: index) {
  %acc0 = tile.fragment_zero {role = "acc"} : !fac
  %va0 = tile.view %a, %r, %k0 {tile.layout = #lay, tile.memory = #memA}
      : (memref<?xf16>, index, index) -> !tile.tile
  %vb0 = tile.view %b, %k0, %c {tile.layout = #lay, tile.memory = #memB}
      : (memref<?xf16>, index, index) -> !tile.tile
  %fa0 = tile.fragment_pack %va0 {role = "a"} : (!tile.tile) -> !fa
  %fb0 = tile.fragment_pack %vb0 {role = "b"} : (!tile.tile) -> !fb
  %m0 = tile.mma %fa0, %fb0, %acc0 : (!fa, !fb, !fac) -> !fac

  %va1 = tile.view %a, %r, %k1 {tile.layout = #lay, tile.memory = #memA}
      : (memref<?xf16>, index, index) -> !tile.tile
  %vb1 = tile.view %b, %k1, %c {tile.layout = #lay, tile.memory = #memB}
      : (memref<?xf16>, index, index) -> !tile.tile
  %fa1 = tile.fragment_pack %va1 {role = "a"} : (!tile.tile) -> !fa
  %fb1 = tile.fragment_pack %vb1 {role = "b"} : (!tile.tile) -> !fb
  %m1 = tile.mma %fa1, %fb1, %m0 : (!fa, !fb, !fac) -> !fac

  %o = tile.fragment_unpack %m1 {role = "acc", tile.layout = #lay}
      : (!fac) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : !tile.tile, memref<?xf32>, index, index
  return
}

// -----

#lay = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                    replica = [] : [] on [], offset = 0>
#memA = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 64>
#memB = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 64>

!fa  = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32",
                      role = "a", layout = "row_major", family = "wmma">
!fb  = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32",
                      role = "b", layout = "col_major", family = "wmma">
!fac = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32",
                      role = "acc", layout = "row_major", family = "wmma">

// 4. NO `role` attribute anywhere — the TYPE carries it.
//
// `FragmentPackOp::verify` returns success without ever reading a `role`
// attribute once the result type is a stated fragment, so this is valid Tile
// IR, and it is the shape a migrated producer (step 3) emits: the redundant
// attribute is exactly what typing was supposed to remove. The lowering read
// `role` off the op and rejected this with ROCM_FRAGMENT_MISSING_CONTRACT.
//
// CHECK-LABEL: func.func @role_comes_from_the_type
// CHECK: tessera_rocm.wmma
// CHECK-SAME: input_dtype = "f16"
func.func @role_comes_from_the_type(%a: memref<?xf16>, %b: memref<?xf16>,
                                    %out: memref<?xf32>, %r: index,
                                    %c: index) {
  %acc = tile.fragment_zero : !fac
  %va = tile.view %a, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : (memref<?xf16>, index, index) -> !tile.tile
  %vb = tile.view %b, %r, %c {tile.layout = #lay, tile.memory = #memB}
      : (memref<?xf16>, index, index) -> !tile.tile
  %fa = tile.fragment_pack %va : (!tile.tile) -> !fa
  %fb = tile.fragment_pack %vb : (!tile.tile) -> !fb
  %m = tile.mma %fa, %fb, %acc : (!fa, !fb, !fac) -> !fac
  %o = tile.fragment_unpack %m {tile.layout = #lay} : (!fac) -> !tile.tile
  tile.store %o, %out, %r, %c {tile.layout = #lay, tile.memory = #memA}
      : !tile.tile, memref<?xf32>, index, index
  return
}
