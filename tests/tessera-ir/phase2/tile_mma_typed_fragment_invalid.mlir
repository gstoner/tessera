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
