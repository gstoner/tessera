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
                       Type storeTy, bool viaTile = false) {
  MLIRContext *ctx = b.getContext();
  int64_t DC = D / 16;
  Type f32 = b.getF32Type();
  Type idxTy = b.getIndexType();
  auto fragTy = VectorType::get({16}, storeTy);
  auto accTy = VectorType::get({8}, f32);
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

  b.setInsertionPointToStart(&f.getBody().front());
  Value Q = f.getArgument(0), Kk = f.getArgument(1), V = f.getArgument(2);
  Value O = f.getArgument(3);
  Value Sq = f.getArgument(4), Sk = f.getArgument(5);
  Value scale = f.getArgument(6), causal = f.getArgument(7);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), c4 = ci(4), c15 = ci(15), c16 = ci(16),
        c32 = ci(32);
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
  Value l15 = b.create<arith::AndIOp>(loc, tid, c15);
  Value half = b.create<arith::ShRUIOp>(loc, tid, c4);
  Value qtile = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bh = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value qbase = b.create<arith::MulIOp>(
      loc, b.create<arith::MulIOp>(loc, bh, Sq), cD);
  Value kbase = b.create<arith::MulIOp>(
      loc, b.create<arith::MulIOp>(loc, bh, Sk), cD);
  Value q0 = b.create<arith::MulIOp>(loc, qtile, c16);

  auto mul = [&](Value x, Value y) { return b.create<arith::MulIOp>(loc, x, y); };
  auto add = [&](Value x, Value y) { return b.create<arith::AddIOp>(loc, x, y); };

  // --- zero sAcc: for i = tid; i < 16*D; i += 32 --- (Q is read from global)
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, c32);
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

  // nKV = (Sk+15)/16 ; lastKt = min(causal ? (q0+15)/16 : nKV-1, nKV-1)
  Value nKV = b.create<arith::DivUIOp>(loc, add(Sk, c15), c16);
  Value nKVm1 = b.create<arith::SubIOp>(loc, nKV, c1);
  Value isCausal = b.create<arith::CmpIOp>(loc, ne, causal, c0);
  Value ckt = b.create<arith::DivUIOp>(loc, add(q0, c15), c16);
  Value lastKt = b.create<arith::SelectOp>(loc, isCausal, ckt, nKVm1);
  Value over = b.create<arith::CmpIOp>(loc, slt, nKVm1, lastKt);
  lastKt = b.create<arith::SelectOp>(loc, over, nKVm1, lastKt);
  Value upper = add(lastKt, c1);

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
  auto wmma = [&](OpBuilder &bb, Location l, Value a, Value bb2, Value acc) {
    OperationState st(l, viaTile ? "tile.mma" : "tessera_rocm.wmma");
    st.addOperands({a, bb2, acc});
    st.addTypes({accTy});
    return bb.create(st)->getResult(0);
  };

  // --- the KV loop ---
  auto kloop = b.create<scf::ForOp>(loc, c0, upper, c1);
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
    Value cs = accZero;
    for (int64_t dc = 0; dc < DC; ++dc) {
      Value dc16 = ci(dc * 16);
      Value aBase = add(add(qbase, mul(qrow_l15, cD)), dc16);
      Value aSafe = b.create<arith::SelectOp>(loc, qrInb, aBase, c0);
      Value aFrag = buildFrag(b, loc, [&](int64_t i) {
        Value v = b.create<memref::LoadOp>(loc, Q, ValueRange{add(aSafe, ci(i))});
        return b.create<arith::SelectOp>(loc, qrInb, v, storeZero);
      });
      Value bBase = add(add(kbase, mul(kr_l15, cD)), dc16);
      Value bSafe = b.create<arith::SelectOp>(loc, krInb, bBase, c0);
      Value bFrag = buildFrag(b, loc, [&](int64_t i) {
        Value v = b.create<memref::LoadOp>(loc, Kk, ValueRange{add(bSafe, ci(i))});
        return b.create<arith::SelectOp>(loc, krInb, v, storeZero);
      });
      cs = wmma(b, loc, aFrag, bFrag, cs);
    }
    // mask + scale -> sS[qi*16 + ki], qi = 2e+half, ki = l15.
    for (int64_t e = 0; e < 8; ++e) {
      Value qi = add(ci(2 * e), half);
      Value gk = add(k0, l15);
      Value csv = b.create<vector::ExtractOp>(loc, cs, ArrayRef<int64_t>{e});
      Value v0 = b.create<arith::MulFOp>(loc, csv, scale);
      Value gkOOB = b.create<arith::CmpIOp>(loc, sge, gk, Sk);
      Value qpos = add(q0, qi);
      Value cmask = b.create<arith::AndIOp>(
          loc, isCausal, b.create<arith::CmpIOp>(loc, slt, qpos, gk));
      Value masked = b.create<arith::OrIOp>(loc, gkOOB, cmask);
      Value v = b.create<arith::SelectOp>(loc, masked, negInf, v0);
      Value sIdx = add(mul(qi, c16), l15);
      b.create<memref::StoreOp>(loc, v, sS, ValueRange{sIdx});
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
        b.create<memref::StoreOp>(loc, p, sS, ValueRange{idx});
        rsum = b.create<arith::AddFOp>(loc, rsum, p);
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
    for (int64_t dc = 0; dc < DC; ++dc) {
      Value dc16 = ci(dc * 16);
      Value pRow = mul(l15, c16);
      Value apFrag = buildFrag(b, loc, [&](int64_t i) {
        Value s = b.create<memref::LoadOp>(loc, sS, ValueRange{add(pRow, ci(i))});
        return b.create<arith::TruncFOp>(loc, storeTy, s);
      });
      Value bvFrag = buildFrag(b, loc, [&](int64_t i) {
        Value kr = add(k0, ci(i));
        Value inb = b.create<arith::CmpIOp>(loc, slt, kr, Sk);
        Value idx = add(add(kbase, mul(kr, cD)), add(dc16, l15));
        Value safe = b.create<arith::SelectOp>(loc, inb, idx, c0);
        Value v = b.create<memref::LoadOp>(loc, V, ValueRange{safe});
        return b.create<arith::SelectOp>(loc, inb, v, storeZero);
      });
      Value cpv = wmma(b, loc, apFrag, bvFrag, accZero);
      for (int64_t e = 0; e < 8; ++e) {
        Value qi = add(ci(2 * e), half);
        Value d = add(dc16, l15);
        Value cpe = b.create<vector::ExtractOp>(loc, cpv, ArrayRef<int64_t>{e});
        Value idx = add(mul(qi, cD), d);
        Value cur = b.create<memref::LoadOp>(loc, sAcc, ValueRange{idx});
        // fused rescale: sAcc = sAcc*corr + cpe (corr = this tile's per-row
        // online-softmax correction, written to scorr by the softmax above).
        Value sc = b.create<memref::LoadOp>(loc, scorr, ValueRange{qi});
        Value resc = b.create<arith::MulFOp>(loc, cur, sc);
        b.create<memref::StoreOp>(loc, b.create<arith::AddFOp>(loc, resc, cpe),
                                  sAcc, ValueRange{idx});
      }
    }
    b.create<gpu::BarrierOp>(loc);
  }

  // O = sAcc / l (final), for i=tid; i<16*D; i+=32.
  b.setInsertionPointAfter(kloop);
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, c32);
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
      if (D <= 0 || D % 16 != 0) {
        op->emitError("generate-wmma-flash-attn-kernel: head_dim must be a "
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
          op->emitError("generate-wmma-flash-attn-kernel: dtype must be f16 or "
                        "bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto abv = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto of = MemRefType::get({ShapedType::kDynamic}, f32);
      // (Q, K, V : memref<?xstore>, O : memref<?xf32>, Sq, Sk : index,
      //  scale : f32, causal : index)
      auto fnTy = b.getFunctionType(
          {abv, abv, abv, of, idxTy, idxTy, f32, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitFlashAttnBody(body, loc, gpuFunc, D, storeTy, viaTile);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMAFlashAttnKernelPass() {
  return std::make_unique<GenerateWMMAFlashAttnKernelPass>();
}
