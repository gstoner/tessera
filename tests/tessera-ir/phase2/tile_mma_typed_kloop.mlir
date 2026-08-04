// RUN: tessera-opt %s | tessera-opt | FileCheck %s
//
// W1.1 step 2 — the canonical K-loop, which the typed form could not express.
//
// This is the fixture the whole item was for. A K-loop accumulator is an
// `scf.for` iter-arg, so inside the body it is a BLOCK ARGUMENT with no
// defining op — and `MMAOp::verify()` recovered the operand contract by chasing
// `getDefiningOp()`, so a typed `tile.mma` reading it could never verify:
//
//     error: 'tile.mma' op accumulator descriptor must match tile.mma
//
// A K-loop around an MMA is what a GEMM *is*, so the typed fragment form was
// unusable by every real GEMM (W1_1_TYPING_DESIGN.md §2). Reading the contract
// from the operand TYPES makes a block argument just another value.
//
// The bare-fragment version of this function is still rejected, and stays in
// `tile_mma_typed_fragment_invalid.mlir` — the legacy path still producer-chases
// until W1.1 step 5 deletes it. The fix is to migrate, not to relax that path.
//
// NOTE (W1.1 step 2b): this VERIFIES. It does not yet LOWER — see the ROCm and
// NVIDIA plan entries under sync key TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03.

#l = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                  replica = [] : [] on [], offset = 0>

// `acc`'s own `elem` IS the accumulator dtype — bf16 storage in, f32
// accumulate. Decision #15a's split, stated in the type where codegen picks the
// instruction rather than in an attribute that stops existing one level down.
!fa  = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "a", layout = "row_major", family = "auto">
!fb  = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "b", layout = "col_major", family = "auto">
!fac = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32",
                      role = "acc", layout = "row_major", family = "auto">

// CHECK-LABEL: func.func @canonical_k_loop
func.func @canonical_k_loop(%a: tensor<16x16xbf16>, %b: tensor<16x16xbf16>,
                            %n: index) -> !fac {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // No `#tile.mma_desc` anywhere in this function. The descriptor is now
  // optional on the typed path — the type carries everything except
  // `k_blocks`, and when a descriptor IS present it must agree.
  // CHECK: tile.fragment_zero
  // CHECK-NOT: mma_desc
  %acc0 = tile.fragment_zero {role = "acc"} : !fac

  // CHECK: scf.for
  // CHECK-SAME: iter_args
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %acc0) -> (!fac) {
    %va = tile.view %a {tile.layout = #l} : (tensor<16x16xbf16>) -> !tile.tile
    %vb = tile.view %b {tile.layout = #l} : (tensor<16x16xbf16>) -> !tile.tile
    %fa = tile.fragment_pack %va {role = "a"} : (!tile.tile) -> !fa
    %fb = tile.fragment_pack %vb {role = "b"} : (!tile.tile) -> !fb

    // The accumulator operand is the loop's block argument. This line is the
    // whole point of W1.1.
    // CHECK: tile.mma
    // CHECK-SAME: role = "acc"
    %res = tile.mma %fa, %fb, %acc : (!fa, !fb, !fac) -> !fac

    // `tile.mma`'s result type equals its accumulator operand's type, which is
    // what lets `scf.for`'s OWN iter-arg/yield equality close the loop. No
    // Tile-specific loop reasoning exists anywhere — that is the payoff of
    // putting the contract in the type rather than in an attribute.
    // CHECK: scf.yield
    scf.yield %res : !fac
  }
  return %r : !fac
}
