//===- GenerateWMMAFlashAttnKernel.cpp - compiler-generated FA-2 fwd ------===//
//
// Expands a `tessera_rocm.flash_attn` directive into a real, fragment-
// materialized RDNA WMMA **flash-attention forward** kernel — the attention
// analog of `generate-wmma-gemm-kernel`. It is a faithful MLIR re-emission of
// the hardware_verified hand-written kernel (tessera_rocm_flash_attn.cpp):
//
//   One wave (32 lanes) per (16-query tile = blockIdx.x, b*h = blockIdx.y).
//   LDS: sQ[16*D] (staged Q), sS[16*16] (scores->probs), sAcc[16*D] (output
//   accumulator), sm/sl/scorr[16] (running max / sum / rescale).
//   Per 16-key tile kt (a runtime scf.for up to the causal/ragged limit):
//     1. S = scale * Q @ K^T   — WMMA accumulate over D/16 head-dim chunks.
//     2. mask (ragged key / causal) -> sS.
//     3. online softmax (lanes 0..15 each own a query row): running max m,
//        rescale corr = exp(m_old - m_new), p = exp(s - m_new), sum l.
//     4. rescale sAcc by corr.
//     5. O += P @ V           — WMMA accumulate over head-dim chunks. Scores are
//        reread from LDS in the P@V A-fragment layout (the layout bridge).
//   Final: O = sAcc / l.
//
// `head_dim` (D, a multiple of 16) is compile-time so the D/16-chunk loops
// unroll; B/H are folded into the grid; Sq/Sk/scale/causal are runtime args.
// `dtype` is the f16/bf16 storage (f32 softmax + accumulate). The score WMMA
// (`tessera_rocm.wmma`) lowers through Stage J to the real `rocdl.wmma`; the
// softmax exp lowers through `convert-math-to-rocdl` (OCML). Validated against a
// numpy attention reference on gfx1151.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"
#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

void emitFlashAttnBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, int64_t D,
                       Type storeTy, bool viaTile = false, bool gqa = false,
                       bool window = false, bool softcap = false,
                       bool dropout = false, bool bias = false,
                       bool twoWave = false, bool saveLse = false) {
  MLIRContext *ctx = b.getContext();
  // Full 16-wide head-dim chunks, plus a ragged remainder. `head_dim` is a
  // COMPILE-TIME attribute here, which is what makes the tail cheap: the
  // element mask in the Q/K fragments below is resolved while emitting, so a
  // ragged D costs no runtime predicate on the contraction side at all.
  int64_t DC = D / 16;
  int64_t DTail = D % 16;
  Type f32 = b.getF32Type();
  Type idxTy = b.getIndexType();
  auto fragTy = VectorType::get({16}, storeTy);
  auto accTy = VectorType::get({8}, f32);
  StringRef fragmentElem = storeTy.isF16() ? "f16" : "bf16";
  auto aFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, 16, fragmentElem, "f32", "a", "row_major", "wmma");
  auto bFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, 16, fragmentElem, "f32", "b", "col_major", "wmma");
  auto accFragmentTy = tessera::tile::FragmentType::get(
      ctx, 16, 16, 16, "f32", "f32", "acc", "row_major", "wmma");
  auto slt = arith::CmpIPredicate::slt;
  auto sge = arith::CmpIPredicate::sge;
  auto ne = arith::CmpIPredicate::ne;

  // Workgroup (LDS) buffers.
  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsT = [&](int64_t n, Type e) {
    return MemRefType::get({n}, e, MemRefLayoutAttrInterface(), ws);
  };
  // Occupancy: Q is NOT staged in LDS — it is read straight from global for the
  // QK^T A-fragment each KV tile. Staging cost 16*D*sizeof(store) of LDS (the
  // 2nd-largest buffer after sAcc), and the kernel is occupancy=LDS-limited
  // (measured: 8 waves/CU @ D=64, 4 @ 128). Dropping sQ frees ~2-4 KB/block →
  // ~+50% resident waves; the extra global Q reloads are hidden by the higher
  // occupancy (the kernel is latency-bound, not bandwidth-bound).
  Value sS = f.addWorkgroupAttribution(ldsT(16 * 16, f32), loc);
  Value sAcc = f.addWorkgroupAttribution(ldsT(16 * D, f32), loc);
  Value sm = f.addWorkgroupAttribution(ldsT(16, f32), loc);
  Value sl = f.addWorkgroupAttribution(ldsT(16, f32), loc);
  Value scorr = f.addWorkgroupAttribution(ldsT(16, f32), loc);
  // G6-B: each wave writes one 16x16 partial QK tile; wave 0 merges the two
  // before the shared online-softmax step.  Bounded overhead: 2 KiB LDS.
  Value sPartial;
  if (twoWave)
    sPartial = f.addWorkgroupAttribution(ldsT(2 * 16 * 16, f32), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value Q = f.getArgument(0), Kk = f.getArgument(1), V = f.getArgument(2);
  Value O = f.getArgument(3);
  Value Sq = f.getArgument(4), Sk = f.getArgument(5);
  Value scale = f.getArgument(6), causal = f.getArgument(7);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), c2 = ci(2), c4 = ci(4), c15 = ci(15),
        c16 = ci(16), c32 = ci(32), c64 = ci(64), cDC = ci(DC);
  Value cD = ci(D), c16D = ci(16 * D);
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));
  Value negInf =
      b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(-1e30f));
  Value storeZero =
      b.create<arith::ConstantOp>(loc, storeTy, b.getFloatAttr(storeTy, 0.0));
  APFloat zAP = cast<FloatAttr>(b.getFloatAttr(storeTy, 0.0)).getValue();
  Value fragZero = b.create<arith::ConstantOp>(
      loc, fragTy, DenseElementsAttr::get(cast<ShapedType>(fragTy), zAP));
  Value accZero = b.create<arith::ConstantOp>(
      loc, accTy, DenseElementsAttr::get(cast<ShapedType>(accTy), APFloat(0.0f)));

  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value lane = twoWave ? b.create<arith::AndIOp>(loc, tid, ci(31)) : tid;
  Value wave = twoWave ? b.create<arith::ShRUIOp>(loc, tid, ci(5)) : c0;
  Value l15 = b.create<arith::AndIOp>(loc, lane, c15);
  Value half = b.create<arith::ShRUIOp>(loc, lane, c4);
  Value qtile = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value qbase = b.create<arith::MulIOp>(
      loc, b.create<arith::MulIOp>(loc, bh, Sq), cD);
  // K/V base. MHA: bh selects the same head for Q and K/V. GQA/MQA: H query
  // heads share G<H key/value heads — query head h reads KV head h/kv_ratio
  // (kv_ratio = H/G; =1 is MHA, =H is MQA). So K/V is [B,G,Sk,D] and the KV
  // block index is b*G + h/kv_ratio. `heads` (H) and `kv_ratio` are runtime args
  // 8/9 in gqa mode.
  Value kvbh = bh;
  if (gqa) {
    Value heads = f.getArgument(8), kvRatio = f.getArgument(9);
    Value bIdx = b.create<arith::DivUIOp>(loc, bh, heads);
    Value hIdx = b.create<arith::RemUIOp>(loc, bh, heads);
    Value gHeads = b.create<arith::DivUIOp>(loc, heads, kvRatio);
    Value kvHead = b.create<arith::DivUIOp>(loc, hIdx, kvRatio);
    kvbh = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, bIdx, gHeads), kvHead);
  }
  Value kbase = b.create<arith::MulIOp>(
      loc, b.create<arith::MulIOp>(loc, kvbh, Sk), cD);
  Value q0 = b.create<arith::MulIOp>(loc, qtile, c16);

  auto mul = [&](Value x, Value y) { return b.create<arith::MulIOp>(loc, x, y); };
  auto add = [&](Value x, Value y) { return b.create<arith::AddIOp>(loc, x, y); };

  // --- zero sAcc: for i = tid; i < 16*D; i += 32 --- (Q is read from global)
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, twoWave ? c64 : c32);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    b.create<memref::StoreOp>(loc, zerof, sAcc,
                              ValueRange{lp.getInductionVar()});
  }
  // if (tid < 16) { sm[tid] = -1e30; sl[tid] = 0; }
  {
    Value lt16 = b.create<arith::CmpIOp>(loc, slt, tid, c16);
    auto ifo = b.create<scf::IfOp>(loc, lt16, /*withElse=*/false);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(ifo.thenBlock());
    b.create<memref::StoreOp>(loc, negInf, sm, ValueRange{tid});
    b.create<memref::StoreOp>(loc, zerof, sl, ValueRange{tid});
  }
  b.create<gpu::BarrierOp>(loc);

  // Sliding-window attention (Mistral-style): query position p attends only to
  // keys in (p - W, p] — a causal band of width W. W is the trailing runtime arg
  // (index `gqa ? 10 : 8`). A windowed kernel is implicitly causal (upper
  // bound), and additionally skips KV tiles entirely below the window's lower
  // edge; the per-element mask trims the boundary tiles.
  Value W;
  if (window)
    W = f.getArgument(gqa ? 10 : 8);
  Value trueI1 = b.create<arith::ConstantIntOp>(loc, 1, /*width=*/1);

  // Gemma-2 logit soft-capping: cap * tanh(S / cap), applied to each scaled
  // score before masking. `cap` is the trailing runtime arg, after the gqa
  // (heads, kv_ratio) and window (W) args when those are present.
  Value cap;
  if (softcap)
    cap = f.getArgument(8 + (gqa ? 2 : 0) + (window ? 1 : 0));

  // Deterministic dropout is applied to P only on the P@V path. The online
  // softmax denominator keeps the undropped probability, matching the scalar
  // semantic reference and allowing backward to replay the same counter.
  Value dropoutP, dropoutSeed;
  if (dropout) {
    int64_t p = 8 + (gqa ? 2 : 0) + (window ? 1 : 0) +
                (softcap ? 1 : 0);
    dropoutP = f.getArgument(p);
    dropoutSeed = f.getArgument(p + 1);
  }

  // Additive attention bias: O = softmax(scale*Q@K^T + attn_bias)*V. The bias
  // memref is the LAST runtime arg (after gqa/window/softcap), f32, host-
  // broadcast to [bh, Sq, Sk] so the kernel indexes bias[(bh*Sq + qpos)*Sk + gk]
  // and adds it to the scaled score after soft-cap and before masking.
  Value biasBuf;
  if (bias)
    biasBuf = f.getArgument(8 + (gqa ? 2 : 0) + (window ? 1 : 0) +
                            (softcap ? 1 : 0) + (dropout ? 2 : 0));

  // nKV = (Sk+15)/16 ; lastKt = min(causal ? (q0+15)/16 : nKV-1, nKV-1)
  Value nKV = b.create<arith::DivUIOp>(loc, add(Sk, c15), c16);
  Value nKVm1 = b.create<arith::SubIOp>(loc, nKV, c1);
  Value isCausal = b.create<arith::CmpIOp>(loc, ne, causal, c0);
  // A windowed kernel is causal for the upper bound regardless of the flag.
  Value useCausal = window ? trueI1 : isCausal;
  Value hasQueryOffset = b.create<arith::CmpIOp>(loc, slt, Sq, Sk);
  Value queryOffset = b.create<arith::SelectOp>(
      loc, hasQueryOffset, b.create<arith::SubIOp>(loc, Sk, Sq), c0);
  Value alignedQ0 = add(q0, queryOffset);
  Value ckt = b.create<arith::DivUIOp>(loc, add(alignedQ0, c15), c16);
  Value lastKt = b.create<arith::SelectOp>(loc, useCausal, ckt, nKVm1);
  Value over = b.create<arith::CmpIOp>(loc, slt, nKVm1, lastKt);
  lastKt = b.create<arith::SelectOp>(loc, over, nKVm1, lastKt);
  Value upper = add(lastKt, c1);

  // Lower KV-tile bound: skip tiles below the window. The oldest key any query
  // in this tile (smallest qpos = q0) attends to is q0 - W + 1; tiles entirely
  // below that contribute only masked (-inf) scores, so we never enter them.
  Value firstKt = c0;
  if (window) {
    Value q0p1 = add(alignedQ0, c1);
    Value above = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                          q0p1, W);
    Value minKey = b.create<arith::SelectOp>(
        loc, above, b.create<arith::SubIOp>(loc, q0p1, W), c0);
    firstKt = b.create<arith::DivUIOp>(loc, minKey, c16);
  }

  // Build a vector<16xstore> fragment from a per-element value lambda.
  auto buildFrag = [&](OpBuilder &bb, Location l,
                       function_ref<Value(int64_t)> elt) {
    Value fr = fragZero;
    for (int64_t i = 0; i < 16; ++i)
      fr = bb.create<vector::InsertOp>(l, elt(i), fr, ArrayRef<int64_t>{i});
    return fr;
  };
  // Fork A (via-tile): emit tile.mma at the Tile-IR seam so the QK^T / P@V
  // matrix ops route through rocm-wave-lds-pipeline + lower-tile-to-rocm (which
  // lowers them back to tessera_rocm.wmma with the same operands). Default emits
  // tessera_rocm.wmma directly. Same operands/types — only the op name differs.
  auto wmma = [&](OpBuilder &bb, Location l, Value a, Value bb2,
                  Value acc) -> Value {
    if (!viaTile) {
      OperationState st(l, "tessera_rocm.wmma");
      st.addOperands({a, bb2, acc});
      st.addTypes({accTy});
      return bb.create(st)->getResult(0);
    }
    // Attention's P fragment is computed in registers/LDS after softmax; it is
    // not a pointer-backed source that can truthfully be reconstructed as a
    // tile.view. Preserve the generator-owned lane map while making the matrix
    // contract typed. Dialect conversion maps these fragment types back to the
    // exact vector ABI, so the unrealized bridges reconcile to identity.
    Value typedA = UnrealizedConversionCastOp::create(bb, l, aFragmentTy, a)
                       .getResult(0);
    Value typedB = UnrealizedConversionCastOp::create(bb, l, bFragmentTy, bb2)
                       .getResult(0);
    Value typedAcc =
        UnrealizedConversionCastOp::create(bb, l, accFragmentTy, acc)
            .getResult(0);
    OperationState st(l, "tile.mma");
    st.addOperands({typedA, typedB, typedAcc});
    st.addTypes({accFragmentTy});
    Value typedResult = bb.create(st)->getResult(0);
    return UnrealizedConversionCastOp::create(bb, l, accTy, typedResult)
        .getResult(0);
  };

  // --- the KV loop (firstKt = window lower bound, else 0) ---
  auto kloop = b.create<scf::ForOp>(loc, firstKt, upper, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(kloop.getBody());
    Value kt = kloop.getInductionVar();
    Value k0 = mul(kt, c16);
    Value kr_l15 = add(k0, l15);
    Value krInb = b.create<arith::CmpIOp>(loc, slt, kr_l15, Sk);

    // S = scale * Q @ K^T over D/16 chunks. Q row = q0 + l15, read from global
    // (no LDS staging — see the sQ note above), masked on the query bound.
    Value qrow_l15 = add(q0, l15);
    Value qrInb = b.create<arith::CmpIOp>(loc, slt, qrow_l15, Sq);
    // `validElems` < 16 marks the ragged head-dim tail: elements at or past it
    // are not loaded at all (the load would read the NEXT row) and enter the
    // fragment as zero. D is the contraction dimension for Q@K^T, so a zero
    // pair contributes exactly nothing to the dot product -- this is exact,
    // not an approximation.
    auto qkChunk = [&](OpBuilder &bb, Value dc16, Value acc,
                       int64_t validElems = 16) {
      Value aBase = add(add(qbase, mul(qrow_l15, cD)), dc16);
      Value aSafe = bb.create<arith::SelectOp>(loc, qrInb, aBase, c0);
      Value aFrag = buildFrag(bb, loc, [&](int64_t i) -> Value {
        if (i >= validElems) return storeZero;
        Value v = bb.create<memref::LoadOp>(loc, Q, ValueRange{add(aSafe, ci(i))});
        return bb.create<arith::SelectOp>(loc, qrInb, v, storeZero);
      });
      Value bBase = add(add(kbase, mul(kr_l15, cD)), dc16);
      Value bSafe = bb.create<arith::SelectOp>(loc, krInb, bBase, c0);
      Value bFrag = buildFrag(bb, loc, [&](int64_t i) -> Value {
        if (i >= validElems) return storeZero;
        Value v = bb.create<memref::LoadOp>(loc, Kk, ValueRange{add(bSafe, ci(i))});
        return bb.create<arith::SelectOp>(loc, krInb, v, storeZero);
      });
      return wmma(bb, loc, aFrag, bFrag, acc);
    };
    Value cs = accZero;
    if (twoWave) {
      auto dloop = b.create<scf::ForOp>(loc, wave, cDC, c2,
                                        ValueRange{accZero});
      {
        OpBuilder::InsertionGuard dg(b);
        b.setInsertionPointToStart(dloop.getBody());
        Value dc16 = mul(dloop.getInductionVar(), c16);
        Value next = qkChunk(b, dc16, dloop.getRegionIterArgs()[0]);
        b.create<scf::YieldOp>(loc, ValueRange{next});
      }
      cs = dloop.getResult(0);
    } else {
      for (int64_t dc = 0; dc < DC; ++dc)
        cs = qkChunk(b, ci(dc * 16), cs);
      if (DTail)
        cs = qkChunk(b, ci(DC * 16), cs, DTail);
    }
    auto emitScore = [&](int64_t e, Value csv) {
      Value qi = add(ci(2 * e), half);
      Value gk = add(k0, l15);
      Value v0 = b.create<arith::MulFOp>(loc, csv, scale);
      Value gkOOB = b.create<arith::CmpIOp>(loc, sge, gk, Sk);
      Value qpos = add(q0, qi);
      // Additive bias is part of the raw logit. The shared recurrence is
      // softcap(scale*QK^T + bias), so combined bias+softcap has one derivative
      // contract on every backend. Guarded on the query bound so masked lanes
      // never read past the [bh,Sq,Sk] buffer; the value is discarded by the
      // -inf select below anyway.
      if (bias) {
        Value qInb = b.create<arith::CmpIOp>(loc, slt, qpos, Sq);
        Value qSafe = b.create<arith::SelectOp>(loc, qInb, qpos, c0);
        Value kSafe = b.create<arith::SelectOp>(loc, gkOOB, c0, gk);
        Value bidx = add(mul(add(mul(bh, Sq), qSafe), Sk), kSafe);
        Value bval = b.create<memref::LoadOp>(loc, biasBuf, ValueRange{bidx});
        v0 = b.create<arith::AddFOp>(loc, v0, bval);
      }
      // Gemma-2 logit soft-cap: cap * tanh((scale*QK^T + bias) / cap).
      if (softcap) {
        Value scaled = b.create<arith::DivFOp>(loc, v0, cap);
        Value t = b.create<math::TanhOp>(loc, scaled);
        v0 = b.create<arith::MulFOp>(loc, cap, t);
      }
      // Causal (future-key) mask — active when causal or windowed.
      Value queryPosition = add(qpos, queryOffset);
      Value cmask = b.create<arith::AndIOp>(
          loc, useCausal,
          b.create<arith::CmpIOp>(loc, slt, queryPosition, gk));
      Value masked = b.create<arith::OrIOp>(loc, gkOOB, cmask);
      // Window lower edge: mask keys older than W (qpos - gk >= W). Only the
      // valid (gk <= qpos) side matters; future keys are already cmask-masked.
      if (window) {
        Value age = b.create<arith::SubIOp>(loc, queryPosition, gk);
        Value tooOld = b.create<arith::CmpIOp>(loc, sge, age, W);
        masked = b.create<arith::OrIOp>(loc, masked, tooOld);
      }
      Value v = b.create<arith::SelectOp>(loc, masked, negInf, v0);
      Value sIdx = add(mul(qi, c16), l15);
      b.create<memref::StoreOp>(loc, v, sS, ValueRange{sIdx});
    };
    // mask + scale -> sS[qi*16 + ki], qi = 2e+half, ki = l15.
    if (twoWave) {
      Value partialBase = mul(wave, ci(16 * 16));
      for (int64_t e = 0; e < 8; ++e) {
        Value qi = add(ci(2 * e), half);
        Value pidx = add(partialBase, add(mul(qi, c16), l15));
        Value csv = b.create<vector::ExtractOp>(loc, cs, ArrayRef<int64_t>{e});
        b.create<memref::StoreOp>(loc, csv, sPartial, ValueRange{pidx});
      }
      b.create<gpu::BarrierOp>(loc);
      Value wave0 = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, wave, c0);
      auto merge = b.create<scf::IfOp>(loc, wave0, /*withElse=*/false);
      {
        OpBuilder::InsertionGuard mg(b);
        b.setInsertionPointToStart(merge.thenBlock());
        for (int64_t e = 0; e < 8; ++e) {
          Value qi = add(ci(2 * e), half);
          Value pidx = add(mul(qi, c16), l15);
          Value p0 = b.create<memref::LoadOp>(loc, sPartial, ValueRange{pidx});
          Value p1 = b.create<memref::LoadOp>(
              loc, sPartial, ValueRange{add(ci(16 * 16), pidx)});
          emitScore(e, b.create<arith::AddFOp>(loc, p0, p1));
        }
      }
    } else {
      for (int64_t e = 0; e < 8; ++e)
        emitScore(e, b.create<vector::ExtractOp>(
                         loc, cs, ArrayRef<int64_t>{e}));
    }
    b.create<gpu::BarrierOp>(loc);

    // online softmax — lanes 0..15 each own query row qi = tid.
    {
      Value lt16 = b.create<arith::CmpIOp>(loc, slt, tid, c16);
      auto ifo = b.create<scf::IfOp>(loc, lt16, /*withElse=*/false);
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(ifo.thenBlock());
      Value qi = tid;
      Value qRow = mul(qi, c16);
      Value rmax = negInf;
      for (int64_t ki = 0; ki < 16; ++ki) {
        Value s = b.create<memref::LoadOp>(loc, sS, ValueRange{add(qRow, ci(ki))});
        rmax = b.create<arith::MaxNumFOp>(loc, rmax, s);
      }
      Value mold = b.create<memref::LoadOp>(loc, sm, ValueRange{qi});
      Value mnew = b.create<arith::MaxNumFOp>(loc, mold, rmax);
      Value moldSmall =
          b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, mold, negInf);
      Value corr0 = b.create<math::ExpOp>(
          loc, b.create<arith::SubFOp>(loc, mold, mnew));
      Value corr = b.create<arith::SelectOp>(loc, moldSmall, zerof, corr0);
      Value rsum = zerof;
      for (int64_t ki = 0; ki < 16; ++ki) {
        Value idx = add(qRow, ci(ki));
        Value s = b.create<memref::LoadOp>(loc, sS, ValueRange{idx});
        Value p = b.create<math::ExpOp>(loc,
                                        b.create<arith::SubFOp>(loc, s, mnew));
        rsum = b.create<arith::AddFOp>(loc, rsum, p);
        Value outputP = p;
        if (dropout) {
          Value qpos = add(q0, qi);
          Value gk = add(k0, ci(ki));
          Value counter = add(mul(add(mul(bh, Sq), qpos), Sk), gk);
          Value counter64 =
              b.create<arith::IndexCastOp>(loc, b.getI64Type(), counter);
          Value hash = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, counter64,
                  b.create<arith::ConstantIntOp>(loc, 1664525, 64)),
              b.create<arith::AddIOp>(
                  loc, dropoutSeed,
                  b.create<arith::ConstantIntOp>(loc, 1013904223, 64)));
          hash = b.create<arith::AndIOp>(
              loc, hash,
              b.create<arith::ConstantIntOp>(loc, 0xffffffffULL, 64));
          Value hash32 =
              b.create<arith::TruncIOp>(loc, b.getI32Type(), hash);
          Value thresholdF = b.create<arith::MulFOp>(
              loc, dropoutP,
              b.create<arith::ConstantOp>(
                  loc, f32, b.getF32FloatAttr(4294967296.0)));
          Value threshold =
              b.create<arith::FPToUIOp>(loc, b.getI32Type(), thresholdF);
          Value keep = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::uge, hash32, threshold);
          Value one =
              b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0));
          Value invKeep = b.create<arith::DivFOp>(
              loc, one, b.create<arith::SubFOp>(loc, one, dropoutP));
          outputP = b.create<arith::SelectOp>(
              loc, keep, b.create<arith::MulFOp>(loc, p, invKeep), zerof);
        }
        b.create<memref::StoreOp>(loc, outputP, sS, ValueRange{idx});
      }
      Value oldl = b.create<memref::LoadOp>(loc, sl, ValueRange{qi});
      Value newl =
          b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, oldl, corr),
                                  rsum);
      b.create<memref::StoreOp>(loc, newl, sl, ValueRange{qi});
      b.create<memref::StoreOp>(loc, mnew, sm, ValueRange{qi});
      b.create<memref::StoreOp>(loc, corr, scorr, ValueRange{qi});
    }
    b.create<gpu::BarrierOp>(loc);

    // O += P @ V over D/16 chunks. The per-row online-softmax correction is
    // FUSED into the accumulator write below (sAcc = sAcc*corr + cpe) instead of
    // a separate rescale pass — saves a full 16*D LDS read+write pass and one
    // barrier per KV tile (each sAcc entry is written exactly once per tile).
    // `tail` marks the ragged head-dim chunk. Here D is the OUTPUT dimension
    // of P@V, not the contraction, so zero-padding is the wrong tool: a lane
    // whose column is past D must write NOTHING. Zeroing its V fragment would
    // still have it store a (correct, zero) value into sAcc at an index one
    // row further on -- silently corrupting the next query's accumulator,
    // which is the LDS analogue of the store-past-the-edge the NVIDIA lane
    // suppresses. So the V read is clamped and the accumulator write is
    // guarded.
    auto emitPVChunk = [&](Value dc16, bool tail = false) {
      Value pRow = mul(l15, c16);
      Value dCol = add(dc16, l15);
      // l15 < DTail  <=>  dc16 + l15 < D  for the tail chunk (dc16 == DC*16).
      Value dInb = tail ? b.create<arith::CmpIOp>(loc, slt, l15, ci(DTail))
                        : Value();
      Value apFrag = buildFrag(b, loc, [&](int64_t i) {
        Value s = b.create<memref::LoadOp>(loc, sS, ValueRange{add(pRow, ci(i))});
        return b.create<arith::TruncFOp>(loc, storeTy, s);
      });
      Value bvFrag = buildFrag(b, loc, [&](int64_t i) {
        Value kr = add(k0, ci(i));
        Value inb = b.create<arith::CmpIOp>(loc, slt, kr, Sk);
        if (dInb) inb = b.create<arith::AndIOp>(loc, inb, dInb);
        Value idx = add(add(kbase, mul(kr, cD)), dCol);
        Value safe = b.create<arith::SelectOp>(loc, inb, idx, c0);
        Value v = b.create<memref::LoadOp>(loc, V, ValueRange{safe});
        return b.create<arith::SelectOp>(loc, inb, v, storeZero);
      });
      Value cpv = wmma(b, loc, apFrag, bvFrag, accZero);
      auto emitAccum = [&](OpBuilder &sb) {
        for (int64_t e = 0; e < 8; ++e) {
          Value qi = add(ci(2 * e), half);
          Value cpe = sb.create<vector::ExtractOp>(loc, cpv, ArrayRef<int64_t>{e});
          Value idx = add(mul(qi, cD), dCol);
          Value cur = sb.create<memref::LoadOp>(loc, sAcc, ValueRange{idx});
          // fused rescale: sAcc = sAcc*corr + cpe (corr = this tile's per-row
          // online-softmax correction, written to scorr by the softmax above).
          Value sc = sb.create<memref::LoadOp>(loc, scorr, ValueRange{qi});
          Value resc = sb.create<arith::MulFOp>(loc, cur, sc);
          sb.create<memref::StoreOp>(loc, sb.create<arith::AddFOp>(loc, resc, cpe),
                                     sAcc, ValueRange{idx});
        }
      };
      if (dInb) {
        auto ifOp = b.create<scf::IfOp>(loc, dInb, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard ig(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        emitAccum(b);
      } else {
        emitAccum(b);
      }
    };
    if (twoWave) {
      auto dloop = b.create<scf::ForOp>(loc, wave, cDC, c2);
      OpBuilder::InsertionGuard pg(b);
      b.setInsertionPointToStart(dloop.getBody());
      emitPVChunk(mul(dloop.getInductionVar(), c16));
    } else {
      for (int64_t dc = 0; dc < DC; ++dc)
        emitPVChunk(ci(dc * 16));
      if (DTail)
        emitPVChunk(ci(DC * 16), /*tail=*/true);
    }
    b.create<gpu::BarrierOp>(loc);
  }

  // O = sAcc / l (final), for i=tid; i<16*D; i+=32.
  b.setInsertionPointAfter(kloop);
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, twoWave ? c64 : c32);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value i = lp.getInductionVar();
    Value r = b.create<arith::DivUIOp>(loc, i, cD);
    Value c = b.create<arith::RemUIOp>(loc, i, cD);
    Value gq = add(q0, r);
    Value inb = b.create<arith::CmpIOp>(loc, slt, gq, Sq);
    auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(ifo.thenBlock());
    Value denom = b.create<memref::LoadOp>(loc, sl, ValueRange{r});
    Value pos =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, denom, zerof);
    Value av = b.create<memref::LoadOp>(loc, sAcc, ValueRange{i});
    Value res = b.create<arith::SelectOp>(
        loc, pos, b.create<arith::DivFOp>(loc, av, denom), zerof);
    Value gidx = add(add(qbase, mul(gq, cD)), c);
    b.create<memref::StoreOp>(loc, res, O, ValueRange{gidx});
  }
  // A training package may bind a launch-owned f32 [B*H,Sq] checkpoint as the
  // final argument. Store the finalized online-softmax statistic once per row.
  if (saveLse) {
    Value checkpoint = f.getArgument(f.getNumArguments() - 1);
    Value lt16 = b.create<arith::CmpIOp>(loc, slt, tid, c16);
    auto ifo = b.create<scf::IfOp>(loc, lt16, /*withElse=*/false);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(ifo.thenBlock());
    Value gq = add(q0, tid);
    auto inBounds = b.create<scf::IfOp>(
        loc, b.create<arith::CmpIOp>(loc, slt, gq, Sq),
        /*withElse=*/false);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(inBounds.thenBlock());
    Value lse = b.create<arith::AddFOp>(
        loc, b.create<memref::LoadOp>(loc, sm, ValueRange{tid}),
        b.create<math::LogOp>(
            loc, b.create<memref::LoadOp>(loc, sl, ValueRange{tid})));
    b.create<memref::StoreOp>(
        loc, lse, checkpoint, ValueRange{add(mul(bh, Sq), gq)});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateWMMAFlashAttnKernelPass
    : PassWrapper<GenerateWMMAFlashAttnKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateWMMAFlashAttnKernelPass)

  // The Option<bool> member is non-copyable; provide the ctors PassWrapper's
  // clonePass() needs (option VALUES are copied separately by MLIR).
  GenerateWMMAFlashAttnKernelPass() = default;
  GenerateWMMAFlashAttnKernelPass(const GenerateWMMAFlashAttnKernelPass &other)
      : PassWrapper(other) {}

  // Fork A: emit tile.mma so the FA matmuls route through the wave/LDS pipeline.
  Option<bool> viaTile{*this, "via-tile",
                       llvm::cl::desc("emit tile.mma (route through the wave/LDS "
                                      "pipeline) instead of tessera_rocm.wmma"),
                       llvm::cl::init(false)};

  StringRef getArgument() const final {
    return "generate-wmma-flash-attn-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.flash_attn directive into a fragment-"
           "materialized RDNA WMMA FA-2 forward gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, vector::VectorDialect,
                    arith::ArithDialect, memref::MemRefDialect,
                    math::MathDialect, tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.flash_attn")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto dAttr = op->getAttrOfType<IntegerAttr>("head_dim");
      if (!nameAttr || !dAttr) {
        op->emitError("tessera_rocm.flash_attn missing name/head_dim");
        return signalPassFailure();
      }
      int64_t D = dAttr.getInt();
      // A ragged head_dim is served by a predicated remainder chunk: the
      // Q@K^T fragments zero-pad past D (exact -- D is the contraction there)
      // and the P@V accumulator write is guarded (D is the OUTPUT dim there,
      // so an unguarded lane would corrupt the next query's row in LDS).
      // Only positivity is required now.
      if (D <= 0) {
        op->emitError("generate-wmma-flash-attn-kernel: head_dim must be "
                      "positive (got ")
            << D << ")";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF16Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f16" && dt != "float16") {
          op->emitError("generate-wmma-flash-attn-kernel: dtype must be f16 or "
                        "bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      // GQA/MQA: grouped query attention — query head h reads KV head
      // h/kv_ratio (kv_ratio = H/G; =1 MHA, =H MQA). The kernel gains two
      // runtime args (heads H, kv_ratio) and reads K/V from the grouped head.
      bool gqa = false;
      if (auto a = op->getAttrOfType<BoolAttr>("gqa"))
        gqa = a.getValue();
      // Sliding-window attention: a causal band of width W (the trailing runtime
      // arg). Composes with gqa (window arg follows heads/kv_ratio).
      bool window = false;
      if (auto a = op->getAttrOfType<BoolAttr>("sliding_window"))
        window = a.getValue();
      // Gemma-2 logit soft-capping: a trailing f32 `cap` runtime arg.
      bool softcap = false;
      if (auto a = op->getAttrOfType<BoolAttr>("logit_softcap"))
        softcap = a.getValue();
      bool dropout = false;
      if (auto a = op->getAttrOfType<BoolAttr>("dropout"))
        dropout = a.getValue();
      // Additive attention bias: a trailing f32 `[bh,Sq,Sk]` memref runtime arg
      // (LAST). O = softmax(scale*Q@K^T + attn_bias)*V.
      bool bias = false;
      if (auto a = op->getAttrOfType<BoolAttr>("attn_bias"))
        bias = a.getValue();
      bool twoWave = false;
      if (auto a = op->getAttrOfType<BoolAttr>("two_wave"))
        twoWave = a.getValue();
      if (twoWave && D != 128) {
        op->emitError("two_wave flash attention currently requires head_dim=128");
        return signalPassFailure();
      }
      bool saveLse = false;
      if (auto a = op->getAttrOfType<BoolAttr>("save_lse"))
        saveLse = a.getValue();

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto abv = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto of = MemRefType::get({ShapedType::kDynamic}, f32);
      // (Q, K, V : memref<?xstore>, O : memref<?xf32>, Sq, Sk : index,
      //  scale : f32, causal : index [, heads : index, kv_ratio : index]
      //  [, window : index])
      SmallVector<Type> argTys{abv, abv, abv, of, idxTy, idxTy, f32, idxTy};
      if (gqa) {
        argTys.push_back(idxTy);  // heads (H)
        argTys.push_back(idxTy);  // kv_ratio (H/G)
      }
      if (window)
        argTys.push_back(idxTy);  // W (sliding-window width)
      if (softcap)
        argTys.push_back(f32);    // cap (Gemma-2 logit soft-cap)
      if (dropout) {
        argTys.push_back(f32);    // dropout probability
        argTys.push_back(b.getI64Type()); // deterministic counter seed
      }
      if (bias)
        argTys.push_back(of);     // attn_bias [bh,Sq,Sk] f32 (LAST)
      if (saveLse)
        argTys.push_back(of);     // finalized LSE checkpoint [bh,Sq]
      auto fnTy = b.getFunctionType(argTys, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitFlashAttnBody(body, loc, gpuFunc, D, storeTy, viaTile, gqa, window,
                        softcap, dropout, bias, twoWave, saveLse);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAFlashAttnKernelPass() {
  return std::make_unique<GenerateWMMAFlashAttnKernelPass>();
}
