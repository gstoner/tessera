// RUN: tessera-opt %s | FileCheck %s
//
// W1.1 step 3 — the typed `tile.mma` fragment form.
//
// `MMAOp::verify()` has enforced this contract for some time, but until now
// **no lit fixture in the tree paired `!tile.fragment` with `tile.mma`**
// (measured 2026-08-02: `grep -rl "tile.mma" tests/tessera-ir/ | xargs grep -l
// "!tile.fragment"` returned nothing). Its typed branch was exercised only by
// MLIR *text* the Python emitters write — so the strongest branch of the Tile
// dialect's most important verifier had no C++-side regression net.
//
// That net has to exist before W1.1 migrates the ten C++ producers onto this
// form, or the migration would be unguarded.

#mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                      a = "bf16", b = "bf16", acc = "f32",
                      a_layout = "row_major", b_layout = "col_major",
                      k_blocks = 1>

// CHECK-LABEL: func.func @typed_mma_fragment_chain
func.func @typed_mma_fragment_chain(%a: tensor<16x16xbf16>,
                                    %b: tensor<16x16xbf16>)
    -> !tile.tile {
  // A fragment is packed from a logical tile view, not straight from a tensor.
  // CHECK: tile.view
  %va = tile.view %a {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>}
      : (tensor<16x16xbf16>) -> !tile.tile
  %vb = tile.view %b {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>}
      : (tensor<16x16xbf16>) -> !tile.tile

  // Operands must come from descriptor-carrying producers with matching roles.
  // CHECK: tile.fragment_pack
  // CHECK-SAME: role = "a"
  %fa = tile.fragment_pack %va {role = "a", mma = #mma}
      : (!tile.tile) -> !tile.fragment
  // CHECK: tile.fragment_pack
  // CHECK-SAME: role = "b"
  %fb = tile.fragment_pack %vb {role = "b", mma = #mma}
      : (!tile.tile) -> !tile.fragment

  // The accumulator starts from a defined zero, not an undef — role "acc".
  // CHECK: tile.fragment_zero
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment

  // Decision #15a in practice: storage is bf16 and the accumulator is f32,
  // and the descriptor is what carries that split down the stack.
  // Three fragment operands in, one fragment out. `acc = "f32"` rides the
  // descriptor on every op in the chain, so assert the mma's shape here rather
  // than re-matching a substring that appears on the pack lines too.
  // CHECK: tile.mma %{{.*}}, %{{.*}}, %{{.*}} {{.*}}(!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
  %res = tile.mma %fa, %fb, %acc {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment

  // The accumulator comes back as a logical tile for the epilogue.
  // CHECK: tile.fragment_unpack
  %out = tile.fragment_unpack %res {role = "acc", mma = #mma, tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>}
      : (!tile.fragment) -> !tile.tile
  return %out : !tile.tile
}

// The case that was IMPOSSIBLE before the data-operand fix.
//
// `WarpSpecLegalityPass` requires a `tile.mma` reading an async-staged tile to
// also read that producer's completion token (WARPSPEC_MMA_NOT_TOKEN_SYNCED),
// so the matrix op is gated on copy completion by SSA rather than program
// order. But `MMAOp::verify()` counted RAW operands, so the token looked like a
// fourth data operand and the typed form rejected it. A producer could satisfy
// the legality pass or the typed verifier, never both — which is why no C++
// producer emitted the typed form at all.
//
// The verifier now counts DATA operands (`tessera::tile::dataOperands`, the
// rule the ROCm lowering already applied privately), so both hold at once.
//
// CHECK-LABEL: func.func @typed_mma_gated_on_a_copy_token
func.func @typed_mma_gated_on_a_copy_token(%a: tensor<16x16xbf16>)
    -> !tile.fragment {
  // CHECK: %[[CP:.*]]:2 = tile.async_copy
  %tile, %tok = tile.async_copy %a
      : (tensor<16x16xbf16>) -> (tensor<16x16xbf16>, !tile.async_token)
  // CHECK: tile.wait_async
  tile.wait_async %tok : (!tile.async_token) -> ()

  %va = tile.view %tile {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>}
      : (tensor<16x16xbf16>) -> !tile.tile
  %fa = tile.fragment_pack %va {role = "a", mma = #mma} : (!tile.tile) -> !tile.fragment
  %fb = tile.fragment_pack %va {role = "b", mma = #mma} : (!tile.tile) -> !tile.fragment
  %acc = tile.fragment_zero {role = "acc", mma = #mma} : !tile.fragment

  // Three DATA operands plus the control token — accepted.
  // CHECK: tile.mma %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {{.*}}!tile.async_token) -> !tile.fragment
  %res = tile.mma %fa, %fb, %acc, %tok {mma = #mma}
      : (!tile.fragment, !tile.fragment, !tile.fragment, !tile.async_token)
      -> !tile.fragment
  return %res : !tile.fragment
}
