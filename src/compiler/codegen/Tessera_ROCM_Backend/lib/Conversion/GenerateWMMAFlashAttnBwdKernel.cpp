//===- GenerateWMMAFlashAttnBwdKernel.cpp - compiler-generated FA-2 bwd ---===//
//
// Expands a `tessera_rocm.flash_attn_bwd` directive into THREE real, fragment-
// materialized RDNA WMMA kernels implementing the textbook FA-2 backward (no
// stored attention matrix — S / P are recomputed per tile):
//
//   <name>_pre  : one wave / (16-query tile, b*h). Scalar pass computing
//                 D[q] = sum_d O[q,d]*dO[q,d] and L[q] = logsumexp_k(scale*QK^T)
//                 (online max/sum). Writes the L / D scratch the matmul kernels
//                 read, so the backward needs nothing saved from the forward.
//   <name>_dkdv : one wave / (16-key tile, b*h). Loops query tiles: recompute
//                 S=scale*Q@K^T, P=exp(S-L) (WMMA over D/16 chunks), dP=dO@V^T
//                 (WMMA), dS=P*(dP-D); accumulate dV += P^T@dO and
//                 dK += scale*dS^T@Q (WMMA, contraction over queries — P / dS
//                 are staged in LDS and reread transposed: the layout bridge).
//   <name>_dq   : one wave / (16-query tile, b*h). Loops key tiles: same
//                 S/P/dP/dS, accumulate dQ += scale*dS@K (WMMA, contraction
//                 over keys; dS reread from LDS in natural layout).
//
// The single hardware primitive used everywhere is the RDNA WMMA
// C[m,n] = sum_k A[m,k]*B[n,k] (A @ B^T), with the 16-wide fragment axis the
// contraction k, m = lane.l15 for A, n = lane.l15 for B, and the C fragment
// holding 8 elements at rows (2*e + lane>>4), col (lane&15) — the same layout
// the forward kernel relies on. `head_dim` (D, a multiple of 16) is compile-
// time so the D/16-chunk loops unroll; B/H fold into the grid; Sq/Sk/scale/
// causal are runtime args. Validated against a numpy attention-backward
// reference (itself checked vs finite differences) on gfx1151.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

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

// Small builder helpers shared by the three kernels.
struct Emit {
  OpBuilder &b;
  Location loc;
  Type f32, idxTy, storeTy;
  VectorType fragTy, accTy;
  Value fragZero, accZero, storeZero, zerof, negInf;
  int64_t D, DC;

  Emit(OpBuilder &b, Location loc, Type storeTy, int64_t D)
      : b(b), loc(loc), storeTy(storeTy), D(D), DC(D / 16) {
    f32 = b.getF32Type();
    idxTy = b.getIndexType();
    fragTy = VectorType::get({16}, storeTy);
    accTy = VectorType::get({8}, f32);
    APFloat zAP = cast<FloatAttr>(b.getFloatAttr(storeTy, 0.0)).getValue();
    storeZero =
        b.create<arith::ConstantOp>(loc, storeTy, b.getFloatAttr(storeTy, 0.0));
    fragZero = b.create<arith::ConstantOp>(
        loc, fragTy, DenseElementsAttr::get(cast<ShapedType>(fragTy), zAP));
    accZero = b.create<arith::ConstantOp>(
        loc, accTy,
        DenseElementsAttr::get(cast<ShapedType>(accTy), APFloat(0.0f)));
    zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));
    negInf = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(-1e30f));
  }

  Value ci(int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); }
  Value add(Value x, Value y) { return b.create<arith::AddIOp>(loc, x, y); }
  Value mul(Value x, Value y) { return b.create<arith::MulIOp>(loc, x, y); }
  Value addf(Value x, Value y) { return b.create<arith::AddFOp>(loc, x, y); }
  Value mulf(Value x, Value y) { return b.create<arith::MulFOp>(loc, x, y); }
  Value subf(Value x, Value y) { return b.create<arith::SubFOp>(loc, x, y); }
  Value sel(Value c, Value a, Value d) {
    return b.create<arith::SelectOp>(loc, c, a, d);
  }
  Value lt(Value a, Value d) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, a, d);
  }
  Value ge(Value a, Value d) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, a, d);
  }
  Value f32load(Value m, Value i) {
    return b.create<memref::LoadOp>(loc, m, ValueRange{i});
  }
  Value toF32(Value v) { return b.create<arith::ExtFOp>(loc, f32, v); }
  Value toStore(Value v) { return b.create<arith::TruncFOp>(loc, storeTy, v); }

  Value buildFrag(function_ref<Value(int64_t)> elt) {
    Value fr = fragZero;
    for (int64_t i = 0; i < 16; ++i)
      fr = b.create<vector::InsertOp>(loc, elt(i), fr, ArrayRef<int64_t>{i});
    return fr;
  }
  Value wmma(Value a, Value bf, Value acc) {
    OperationState st(loc, "tessera_rocm.wmma");
    st.addOperands({a, bf, acc});
    st.addTypes({accTy});
    return b.create(st)->getResult(0);
  }
  Value ext(Value vec, int64_t e) {
    return b.create<vector::ExtractOp>(loc, vec, ArrayRef<int64_t>{e});
  }
};

using gpu::AddressSpace;
static MemRefType ldsT(MLIRContext *ctx, int64_t n, Type e) {
  auto ws = gpu::AddressSpaceAttr::get(ctx, AddressSpace::Workgroup);
  return MemRefType::get({n}, e, MemRefLayoutAttrInterface(), ws);
}

//===----------------------------------------------------------------------===//
// <name>_pre : scalar L (logsumexp) + D (rowsum O*dO) per query row.
//   args: (Q, K, dO : memref<?xstore>, O, L, Dd : memref<?xf32>,
//          Sq, Sk : index, scale : f32, causal : index)
//===----------------------------------------------------------------------===//
void emitPre(OpBuilder &b, Location loc, gpu::GPUFuncOp f, int64_t D,
             Type storeTy) {
  b.setInsertionPointToStart(&f.getBody().front());
  Emit e(b, loc, storeTy, D);
  Value Q = f.getArgument(0), Kk = f.getArgument(1), dO = f.getArgument(2);
  Value O = f.getArgument(3), L = f.getArgument(4), Dd = f.getArgument(5);
  Value Sq = f.getArgument(6), Sk = f.getArgument(7);
  Value scale = f.getArgument(8), causal = f.getArgument(9);

  Value c0 = e.ci(0), c1 = e.ci(1), c16 = e.ci(16), cD = e.ci(D);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value qtile = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value q0 = e.mul(qtile, c16);
  Value qbase = e.mul(e.mul(bh, Sq), cD);
  Value kbase = e.mul(e.mul(bh, Sk), cD);
  Value isCausal = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                           causal, c0);

  // lanes 0..15 each own one query row.
  Value lt16 = e.lt(tid, c16);
  auto ifo = b.create<scf::IfOp>(loc, lt16, /*withElse=*/false);
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(ifo.thenBlock());
  Value gq = e.add(q0, tid);
  Value inb = e.lt(gq, Sq);
  Value gqSafe = e.sel(inb, gq, c0);
  Value rowBase = e.add(qbase, e.mul(gqSafe, cD));

  // D[q] = sum_d O[q,d]*dO[q,d]
  auto dloop = b.create<scf::ForOp>(loc, c0, cD, c1, ValueRange{e.zerof});
  {
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(dloop.getBody());
    Value d = dloop.getInductionVar();
    Value acc = dloop.getRegionIterArg(0);
    Value idx = e.add(rowBase, d);
    Value ov = e.f32load(O, idx);
    Value dov = e.toF32(b.create<memref::LoadOp>(loc, dO, ValueRange{idx}));
    b.create<scf::YieldOp>(loc, ValueRange{e.addf(acc, e.mulf(ov, dov))});
  }
  Value Dq = dloop.getResult(0);

  // L[q] = logsumexp_k(scale * Q@K^T), online max/sum over keys.
  auto kloop = b.create<scf::ForOp>(loc, c0, Sk, c1,
                                    ValueRange{e.negInf, e.zerof});
  {
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(kloop.getBody());
    Value k = kloop.getInductionVar();
    Value m = kloop.getRegionIterArg(0), l = kloop.getRegionIterArg(1);
    Value kRow = e.add(kbase, e.mul(k, cD));
    // s = scale * sum_d Q[gq,d]*K[k,d]
    auto sloop = b.create<scf::ForOp>(loc, c0, cD, c1, ValueRange{e.zerof});
    {
      OpBuilder::InsertionGuard g3(b);
      b.setInsertionPointToStart(sloop.getBody());
      Value d = sloop.getInductionVar();
      Value acc = sloop.getRegionIterArg(0);
      Value qv = e.toF32(
          b.create<memref::LoadOp>(loc, Q, ValueRange{e.add(rowBase, d)}));
      Value kv = e.toF32(
          b.create<memref::LoadOp>(loc, Kk, ValueRange{e.add(kRow, d)}));
      b.create<scf::YieldOp>(loc, ValueRange{e.addf(acc, e.mulf(qv, kv))});
    }
    Value s = e.mulf(sloop.getResult(0), scale);
    // causal: key k > query gq -> mask out (s = -inf).
    Value masked = b.create<arith::AndIOp>(loc, isCausal, e.lt(gq, k));
    s = e.sel(masked, e.negInf, s);
    Value mnew = b.create<arith::MaxNumFOp>(loc, m, s);
    Value corr = b.create<math::ExpOp>(loc, e.subf(m, mnew));
    Value pl = b.create<math::ExpOp>(loc, e.subf(s, mnew));
    Value lnew = e.addf(e.mulf(l, corr), pl);
    b.create<scf::YieldOp>(loc, ValueRange{mnew, lnew});
  }
  Value mfin = kloop.getResult(0), lfin = kloop.getResult(1);
  Value Lq = e.addf(mfin, b.create<math::LogOp>(loc, lfin));

  auto wif = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(wif.thenBlock());
    Value gidx = e.add(e.mul(bh, Sq), gq);
    b.create<memref::StoreOp>(loc, Lq, L, ValueRange{gidx});
    b.create<memref::StoreOp>(loc, Dq, Dd, ValueRange{gidx});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

//===----------------------------------------------------------------------===//
// Shared inner block for the two WMMA kernels: recompute the 16x16 score tile
// for (query tile q0, key tile k0), producing P (and dS) in LDS.  Each lane
// owns the 8 (q=2e+half, k=l15) score elements.  Writes:
//   sP[q*16+k]  = P  (store type)  -- only if wantP
//   sDS[q*16+k] = dS (store type)
//===----------------------------------------------------------------------===//
struct ScoreCtx {
  Value Q, Kk, V, dO, L, Dd;
  Value Sq, Sk, scale, isCausal;
  Value qbase, kbase, q0, k0, bh;
  Value l15, half;
};

void recomputeScoreTile(Emit &e, OpBuilder &b, Location loc, const ScoreCtx &x,
                        Value sP, Value sDS, bool wantP) {
  Value cD = e.ci(e.D), c0 = e.ci(0), c16 = e.ci(16), c1 = e.ci(1);
  Value krow = e.add(x.k0, x.l15);       // this lane's key (n axis of S)
  Value krInb = e.lt(krow, x.Sk);
  Value qrow_l = e.add(x.q0, x.l15);     // this lane's query (m axis of S)
  Value qrInb = e.lt(qrow_l, x.Sq);

  // S = scale * Q@K^T  and  dP = dO@V^T  over D/16 head-dim chunks.
  Value cs = e.accZero, cp = e.accZero;
  for (int64_t dc = 0; dc < e.DC; ++dc) {
    Value dc16 = e.ci(dc * 16);
    Value qB = e.add(e.add(x.qbase, e.mul(qrow_l, cD)), dc16);
    Value qSafe = e.sel(qrInb, qB, c0);
    Value kB = e.add(e.add(x.kbase, e.mul(krow, cD)), dc16);
    Value kSafe = e.sel(krInb, kB, c0);
    Value aQ = e.buildFrag([&](int64_t i) {
      Value v = b.create<memref::LoadOp>(loc, x.Q, ValueRange{e.add(qSafe, e.ci(i))});
      return e.sel(qrInb, v, e.storeZero);
    });
    Value bK = e.buildFrag([&](int64_t i) {
      Value v = b.create<memref::LoadOp>(loc, x.Kk, ValueRange{e.add(kSafe, e.ci(i))});
      return e.sel(krInb, v, e.storeZero);
    });
    cs = e.wmma(aQ, bK, cs);
    Value adO = e.buildFrag([&](int64_t i) {
      Value v = b.create<memref::LoadOp>(loc, x.dO, ValueRange{e.add(qSafe, e.ci(i))});
      return e.sel(qrInb, v, e.storeZero);
    });
    Value bV = e.buildFrag([&](int64_t i) {
      Value v = b.create<memref::LoadOp>(loc, x.V, ValueRange{e.add(kSafe, e.ci(i))});
      return e.sel(krInb, v, e.storeZero);
    });
    cp = e.wmma(adO, bV, cp);
  }

  // For each of this lane's 8 elements: q = 2e+half, k = l15.
  Value gk = e.add(x.k0, x.l15);
  for (int64_t el = 0; el < 8; ++el) {
    Value qi = e.add(e.ci(2 * el), x.half);
    Value gqi = e.add(x.q0, qi);
    Value gqSafe = e.sel(e.lt(gqi, x.Sq), gqi, c0);
    Value Lidx = e.add(e.mul(x.bh, x.Sq), gqSafe);
    Value Lq = e.f32load(x.L, Lidx);
    Value Dq = e.f32load(x.Dd, Lidx);
    Value s = e.mulf(e.ext(cs, el), x.scale);
    Value P = b.create<math::ExpOp>(loc, e.subf(s, Lq));
    // mask: query OOB, key OOB, or causal (key > query) -> P = 0
    Value m1 = e.ge(gqi, x.Sq);
    Value m2 = e.ge(gk, x.Sk);
    Value m3 = b.create<arith::AndIOp>(loc, x.isCausal, e.lt(gqi, gk));
    Value masked = b.create<arith::OrIOp>(
        loc, b.create<arith::OrIOp>(loc, m1, m2), m3);
    P = e.sel(masked, e.zerof, P);
    Value dS = e.mulf(P, e.subf(e.ext(cp, el), Dq));
    dS = e.sel(masked, e.zerof, dS);
    Value sIdx = e.add(e.mul(qi, c16), x.l15);
    if (wantP)
      b.create<memref::StoreOp>(loc, e.toStore(P), sP, ValueRange{sIdx});
    b.create<memref::StoreOp>(loc, e.toStore(dS), sDS, ValueRange{sIdx});
  }
}

//===----------------------------------------------------------------------===//
// <name>_dkdv : per (16-key tile, b*h); loop query tiles; dK, dV.
//   args: (Q, K, V, dO : store, L, Dd : f32, dK, dV : f32,
//          Sq, Sk : index, scale : f32, causal : index)
//===----------------------------------------------------------------------===//
void emitDkDv(OpBuilder &b, Location loc, gpu::GPUFuncOp f, int64_t D,
              Type storeTy) {
  MLIRContext *ctx = b.getContext();
  Value sP = f.addWorkgroupAttribution(ldsT(ctx, 16 * 16, storeTy), loc);
  Value sDS = f.addWorkgroupAttribution(ldsT(ctx, 16 * 16, storeTy), loc);
  Value dKacc = f.addWorkgroupAttribution(ldsT(ctx, 16 * D, b.getF32Type()), loc);
  Value dVacc = f.addWorkgroupAttribution(ldsT(ctx, 16 * D, b.getF32Type()), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Emit e(b, loc, storeTy, D);
  Value Q = f.getArgument(0), Kk = f.getArgument(1), V = f.getArgument(2);
  Value dO = f.getArgument(3), L = f.getArgument(4), Dd = f.getArgument(5);
  Value dK = f.getArgument(6), dV = f.getArgument(7);
  Value Sq = f.getArgument(8), Sk = f.getArgument(9);
  Value scale = f.getArgument(10), causal = f.getArgument(11);

  Value c0 = e.ci(0), c1 = e.ci(1), c15 = e.ci(15), c16 = e.ci(16),
        c32 = e.ci(32), cD = e.ci(D), c16D = e.ci(16 * D);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value l15 = b.create<arith::AndIOp>(loc, tid, c15);
  Value half = b.create<arith::ShRUIOp>(loc, tid, e.ci(4));
  Value ktile = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value k0 = e.mul(ktile, c16);
  Value qbase = e.mul(e.mul(bh, Sq), cD);
  Value kbase = e.mul(e.mul(bh, Sk), cD);
  Value isCausal = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                           causal, c0);

  // zero dK/dV accumulators.
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, c32);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    b.create<memref::StoreOp>(loc, e.zerof, dKacc,
                              ValueRange{lp.getInductionVar()});
    b.create<memref::StoreOp>(loc, e.zerof, dVacc,
                              ValueRange{lp.getInductionVar()});
  }
  b.create<gpu::BarrierOp>(loc);

  Value nQ = b.create<arith::DivUIOp>(loc, e.add(Sq, c15), c16);
  auto qloop = b.create<scf::ForOp>(loc, c0, nQ, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(qloop.getBody());
    Value q0 = e.mul(qloop.getInductionVar(), c16);

    ScoreCtx x{Q, Kk, V, dO, L, Dd, Sq, Sk, scale, isCausal,
               qbase, kbase, q0, k0, bh, l15, half};
    recomputeScoreTile(e, b, loc, x, sP, sDS, /*wantP=*/true);
    b.create<gpu::BarrierOp>(loc);

    // dV += P^T @ dO ; dK += scale * dS^T @ Q  (contraction over queries).
    for (int64_t dc = 0; dc < e.DC; ++dc) {
      Value dc16 = e.ci(dc * 16);
      Value aP = e.buildFrag([&](int64_t i) {
        return b.create<memref::LoadOp>(
            loc, sP, ValueRange{e.add(e.mul(e.ci(i), c16), l15)});
      });
      Value aDS = e.buildFrag([&](int64_t i) {
        return b.create<memref::LoadOp>(
            loc, sDS, ValueRange{e.add(e.mul(e.ci(i), c16), l15)});
      });
      Value bdO = e.buildFrag([&](int64_t i) {
        Value gq = e.add(q0, e.ci(i));
        Value inb = e.lt(gq, Sq);
        Value idx = e.add(e.add(e.add(qbase, e.mul(gq, cD)), dc16), l15);
        return e.sel(inb,
                     b.create<memref::LoadOp>(
                         loc, dO, ValueRange{e.sel(inb, idx, c0)}),
                     e.storeZero);
      });
      Value bQ = e.buildFrag([&](int64_t i) {
        Value gq = e.add(q0, e.ci(i));
        Value inb = e.lt(gq, Sq);
        Value idx = e.add(e.add(e.add(qbase, e.mul(gq, cD)), dc16), l15);
        return e.sel(inb,
                     b.create<memref::LoadOp>(
                         loc, Q, ValueRange{e.sel(inb, idx, c0)}),
                     e.storeZero);
      });
      Value rV = e.wmma(aP, bdO, e.accZero);
      Value rK = e.wmma(aDS, bQ, e.accZero);
      for (int64_t el = 0; el < 8; ++el) {
        Value krow = e.add(e.ci(2 * el), half);      // key within tile
        Value d = e.add(dc16, l15);
        Value idx = e.add(e.mul(krow, cD), d);
        Value curV = e.f32load(dVacc, idx);
        b.create<memref::StoreOp>(loc, e.addf(curV, e.ext(rV, el)), dVacc,
                                  ValueRange{idx});
        Value curK = e.f32load(dKacc, idx);
        b.create<memref::StoreOp>(
            loc, e.addf(curK, e.mulf(e.ext(rK, el), scale)), dKacc,
            ValueRange{idx});
      }
    }
    b.create<gpu::BarrierOp>(loc);
  }

  // write dK/dV accumulators to global for keys k0+row < Sk.
  b.setInsertionPointAfter(qloop);
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, c32);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value i = lp.getInductionVar();
    Value r = b.create<arith::DivUIOp>(loc, i, cD);
    Value c = b.create<arith::RemUIOp>(loc, i, cD);
    Value gk = e.add(k0, r);
    Value inb = e.lt(gk, Sk);
    auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(ifo.thenBlock());
    Value gidx = e.add(e.add(kbase, e.mul(gk, cD)), c);
    b.create<memref::StoreOp>(loc, e.f32load(dKacc, i), dK, ValueRange{gidx});
    b.create<memref::StoreOp>(loc, e.f32load(dVacc, i), dV, ValueRange{gidx});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

//===----------------------------------------------------------------------===//
// <name>_dq : per (16-query tile, b*h); loop key tiles; dQ.
//   args: (Q, K, V, dO : store, L, Dd : f32, dQ : f32,
//          Sq, Sk : index, scale : f32, causal : index)
//===----------------------------------------------------------------------===//
void emitDq(OpBuilder &b, Location loc, gpu::GPUFuncOp f, int64_t D,
            Type storeTy) {
  MLIRContext *ctx = b.getContext();
  Value sDS = f.addWorkgroupAttribution(ldsT(ctx, 16 * 16, storeTy), loc);
  Value dQacc = f.addWorkgroupAttribution(ldsT(ctx, 16 * D, b.getF32Type()), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Emit e(b, loc, storeTy, D);
  Value Q = f.getArgument(0), Kk = f.getArgument(1), V = f.getArgument(2);
  Value dO = f.getArgument(3), L = f.getArgument(4), Dd = f.getArgument(5);
  Value dQ = f.getArgument(6);
  Value Sq = f.getArgument(7), Sk = f.getArgument(8);
  Value scale = f.getArgument(9), causal = f.getArgument(10);

  Value c0 = e.ci(0), c1 = e.ci(1), c15 = e.ci(15), c16 = e.ci(16),
        c32 = e.ci(32), cD = e.ci(D), c16D = e.ci(16 * D);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value l15 = b.create<arith::AndIOp>(loc, tid, c15);
  Value half = b.create<arith::ShRUIOp>(loc, tid, e.ci(4));
  Value qtile = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value q0 = e.mul(qtile, c16);
  Value qbase = e.mul(e.mul(bh, Sq), cD);
  Value kbase = e.mul(e.mul(bh, Sk), cD);
  Value isCausal = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                           causal, c0);

  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, c32);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    b.create<memref::StoreOp>(loc, e.zerof, dQacc,
                              ValueRange{lp.getInductionVar()});
  }
  b.create<gpu::BarrierOp>(loc);

  // Iterate every key tile; the per-element causal/OOB mask in
  // recomputeScoreTile zeroes the contributions beyond the diagonal (a causal
  // loop-bound is a future perf optimization, deliberately not risked here).
  Value nK = b.create<arith::DivUIOp>(loc, e.add(Sk, c15), c16);
  auto kloop = b.create<scf::ForOp>(loc, c0, nK, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(kloop.getBody());
    Value k0 = e.mul(kloop.getInductionVar(), c16);

    ScoreCtx x{Q, Kk, V, dO, L, Dd, Sq, Sk, scale, isCausal,
               qbase, kbase, q0, k0, bh, l15, half};
    recomputeScoreTile(e, b, loc, x, /*sP=*/Value(), sDS, /*wantP=*/false);
    b.create<gpu::BarrierOp>(loc);

    // dQ += scale * dS @ K  (contraction over keys; dS natural layout).
    for (int64_t dc = 0; dc < e.DC; ++dc) {
      Value dc16 = e.ci(dc * 16);
      Value aDS = e.buildFrag([&](int64_t i) {
        return b.create<memref::LoadOp>(
            loc, sDS, ValueRange{e.add(e.mul(l15, c16), e.ci(i))});
      });
      Value bK = e.buildFrag([&](int64_t i) {
        Value gk = e.add(k0, e.ci(i));
        Value inb = e.lt(gk, Sk);
        Value idx = e.add(e.add(e.add(kbase, e.mul(gk, cD)), dc16), l15);
        return e.sel(inb,
                     b.create<memref::LoadOp>(
                         loc, Kk, ValueRange{e.sel(inb, idx, c0)}),
                     e.storeZero);
      });
      Value rQ = e.wmma(aDS, bK, e.accZero);
      for (int64_t el = 0; el < 8; ++el) {
        Value qi = e.add(e.ci(2 * el), half);
        Value d = e.add(dc16, l15);
        Value idx = e.add(e.mul(qi, cD), d);
        Value cur = e.f32load(dQacc, idx);
        b.create<memref::StoreOp>(
            loc, e.addf(cur, e.mulf(e.ext(rQ, el), scale)), dQacc,
            ValueRange{idx});
      }
    }
    b.create<gpu::BarrierOp>(loc);
  }

  b.setInsertionPointAfter(kloop);
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, c32);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value i = lp.getInductionVar();
    Value r = b.create<arith::DivUIOp>(loc, i, cD);
    Value c = b.create<arith::RemUIOp>(loc, i, cD);
    Value gq = e.add(q0, r);
    Value inb = e.lt(gq, Sq);
    auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(ifo.thenBlock());
    Value gidx = e.add(e.add(qbase, e.mul(gq, cD)), c);
    b.create<memref::StoreOp>(loc, e.f32load(dQacc, i), dQ, ValueRange{gidx});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateWMMAFlashAttnBwdKernelPass
    : PassWrapper<GenerateWMMAFlashAttnBwdKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateWMMAFlashAttnBwdKernelPass)

  StringRef getArgument() const final {
    return "generate-wmma-flash-attn-bwd-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.flash_attn_bwd directive into the three "
           "fragment-materialized RDNA WMMA FA-2 backward gpu kernels "
           "(_pre/_dkdv/_dq; compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, vector::VectorDialect,
                    arith::ArithDialect, memref::MemRefDialect,
                    math::MathDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.flash_attn_bwd")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto dAttr = op->getAttrOfType<IntegerAttr>("head_dim");
      if (!nameAttr || !dAttr) {
        op->emitError("tessera_rocm.flash_attn_bwd missing name/head_dim");
        return signalPassFailure();
      }
      int64_t D = dAttr.getInt();
      if (D <= 0 || D % 16 != 0) {
        op->emitError("generate-wmma-flash-attn-bwd-kernel: head_dim must be a "
                      "positive multiple of 16 (got ")
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
          op->emitError("generate-wmma-flash-attn-bwd-kernel: dtype must be "
                        "f16 or bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto sv = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto fv = MemRefType::get({ShapedType::kDynamic}, f32);

      auto mk = [&](StringRef suffix, ArrayRef<Type> args,
                    function_ref<void(OpBuilder &, Location, gpu::GPUFuncOp)> body) {
        auto fnTy = b.getFunctionType(args, {});
        auto fn = b.create<gpu::GPUFuncOp>(loc, kname + suffix.str(), fnTy);
        fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
        OpBuilder bb(fn.getContext());
        body(bb, loc, fn);
      };

      // _pre : (Q,K,dO:store, O,L,Dd:f32, Sq,Sk:idx, scale:f32, causal:idx)
      mk("_pre", {sv, sv, sv, fv, fv, fv, idxTy, idxTy, f32, idxTy},
         [&](OpBuilder &bb, Location l, gpu::GPUFuncOp fn) {
           emitPre(bb, l, fn, D, storeTy);
         });
      // _dkdv : (Q,K,V,dO:store, L,Dd:f32, dK,dV:f32, Sq,Sk:idx, scale, causal)
      mk("_dkdv", {sv, sv, sv, sv, fv, fv, fv, fv, idxTy, idxTy, f32, idxTy},
         [&](OpBuilder &bb, Location l, gpu::GPUFuncOp fn) {
           emitDkDv(bb, l, fn, D, storeTy);
         });
      // _dq : (Q,K,V,dO:store, L,Dd:f32, dQ:f32, Sq,Sk:idx, scale, causal)
      mk("_dq", {sv, sv, sv, sv, fv, fv, fv, idxTy, idxTy, f32, idxTy},
         [&](OpBuilder &bb, Location l, gpu::GPUFuncOp fn) {
           emitDq(bb, l, fn, D, storeTy);
         });
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAFlashAttnBwdKernelPass() {
  return std::make_unique<GenerateWMMAFlashAttnBwdKernelPass>();
}
