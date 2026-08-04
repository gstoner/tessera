// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
//
// W1.1 step 3 — the typed `tile.mma` contract must REJECT.
//
// This is the regression net the producer migration needs. Each case below is
// a way a migrating producer could get the typed form subtly wrong while still
// emitting text that contains "tile.mma" and "!tile.fragment" — i.e. exactly
// what a substring assertion would wave through.

#mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                      a = "bf16", b = "bf16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
#layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                       replica = [] : [] on [], offset = 0>

// The B operand carries role "a" — a transposed-operand bug that produces a
// numerically wrong matmul, not a crash.
func.func @swapped_operand_roles(%a: tensor<16x16xbf16>) -> !tile.fragment {
  %va = tile.view %a {tile.layout = #layout} : (tensor<16x16xbf16>) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma} : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %va {role = "a", mma = #mma} : (!tile.tile) -> !tile.fragment
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment
  // CHECK: error: 'tile.mma' op expects a fragment with role "b"
  %res = tile.mma %fa, %fb, %acc {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  return %res : !tile.fragment
}

// -----

#mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                      a = "bf16", b = "bf16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
// A DIFFERENT instruction shape — the operands were packed for k = 8.
#mma_k8 = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 8,
                         a = "bf16", b = "bf16", acc = "f32",
                         a_layout = "row_major", b_layout = "col_major",
                         k_blocks = 1>
#layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                       replica = [] : [] on [], offset = 0>

// Fragments packed for one instruction shape fed to an mma declaring another.
// The register layouts genuinely differ, so this is a real miscompile.
func.func @mismatched_descriptor(%a: tensor<16x16xbf16>) -> !tile.fragment {
  %va = tile.view %a {tile.layout = #layout} : (tensor<16x16xbf16>) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma_k8} : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %va {role = "b", mma = #mma_k8} : (!tile.tile) -> !tile.fragment
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment
  // CHECK: error: 'tile.mma' op fragment descriptor must match tile.mma
  %res = tile.mma %fa, %fb, %acc {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  return %res : !tile.fragment
}

// -----

#mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                      a = "bf16", b = "bf16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>
#layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                       replica = [] : [] on [], offset = 0>

// A typed mma with no accumulator: the non-NVFP4 form is exactly A, B, acc.
func.func @typed_form_missing_accumulator(%a: tensor<16x16xbf16>) -> !tile.fragment {
  %va = tile.view %a {tile.layout = #layout} : (tensor<16x16xbf16>) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma} : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %va {role = "b", mma = #mma} : (!tile.tile) -> !tile.fragment
  // CHECK: error: 'tile.mma' op typed fragment form expects A, B, accumulator -> !tile.fragment
  %res = tile.mma %fa, %fb {mma = #mma}
      : (!tile.fragment, !tile.fragment) -> !tile.fragment
  return %res : !tile.fragment
}

// -----

#layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                       replica = [] : [] on [], offset = 0>

// Fragments present but no descriptor on the mma: the typed form cannot be
// lowered to a physical cooperative-matrix instruction without one.
func.func @typed_form_without_descriptor(%a: tensor<16x16xbf16>) -> !tile.fragment {
  %va = tile.view %a {tile.layout = #layout} : (tensor<16x16xbf16>) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a"} : (!tile.tile) -> !tile.fragment
  // CHECK: error
  %res = tile.mma %fa, %fa, %fa : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  return %res : !tile.fragment
}

// -----

#kl = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                     a = "bf16", b = "bf16", acc = "f32",
                     a_layout = "row_major", b_layout = "col_major",
                     k_blocks = 1>
#kllayout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"],
                         replica = [] : [] on [], offset = 0>

// ── A LIMITATION, not a contract. This case must FLIP to accepting. ─────────
//
// The canonical K-loop: the accumulator is an `scf.for` iter_arg, so inside the
// body it is a BLOCK ARGUMENT. `MMAOp::verify()` recovers the contract by
// chasing `accumulator.getDefiningOp()`, and a block argument has none, so the
// descriptor-equality check can never pass.
//
// A K-loop accumulator is a block argument by construction and a K-loop around
// an MMA is what a GEMM IS — so the typed fragment form is currently unusable
// by every real GEMM. That, better than the (since-fixed) warp-spec token
// conflict, explains why no C++ producer emits the typed form:
// W1_1_TYPING_DESIGN.md §2.
//
// It is recorded here so the limitation is visible and has a forcing function.
// **When W1.1 step 2 parameterizes `!tile.fragment` on (m, n, k, elem, acc,
// role, layout), this stops being an error** — the iter_arg carries its own
// contract and `scf.for`'s iter-arg/yield type equality enforces the
// loop-carried half for free. At that point MOVE this function into
// tile_mma_typed_fragment.mlir. Do not "fix" it by relaxing the accumulator
// check: that would drop the contract for the straight-line case too.
func.func @kloop_accumulator_is_a_block_argument(
    %a: tensor<16x16xbf16>, %b: tensor<16x16xbf16>, %n: index)
    -> !tile.fragment {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc0 = tile.fragment_zero {role = "acc", mma = #kl} : !tile.fragment
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %acc0)
      -> (!tile.fragment) {
    %va = tile.view %a {tile.layout = #kllayout} : (tensor<16x16xbf16>) -> !tile.tile
    %vb = tile.view %b {tile.layout = #kllayout} : (tensor<16x16xbf16>) -> !tile.tile
    %fa = tile.fragment_pack %va {role = "a", mma = #kl} : (!tile.tile) -> !tile.fragment
    %fb = tile.fragment_pack %vb {role = "b", mma = #kl} : (!tile.tile) -> !tile.fragment
    // CHECK: error: 'tile.mma' op accumulator descriptor must match tile.mma
    %res = tile.mma %fa, %fb, %acc {mma = #kl}
        : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
    scf.yield %res : !tile.fragment
  }
  return %r : !tile.fragment
}
