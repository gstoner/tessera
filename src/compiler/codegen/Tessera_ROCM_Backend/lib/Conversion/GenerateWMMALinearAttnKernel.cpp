//===- GenerateWMMALinearAttnKernel.cpp - compiler-generated linear attn --===//
//
// Expands a `tessera_rocm.linear_attn` directive into a real, fragment-
// materialized RDNA WMMA **linear-attention forward** kernel — the quadratic-
// parallel form `O = (φ(Q) φ(K)ᵀ ⊙ causal) @ V`. It is structurally flash
// attention forward MINUS the online softmax: there is no running max / exp /
// normalization, the score matrix is masked **multiplicatively** (masked → 0,
// not −∞), and `O += A @ V` accumulates directly with no final divide.
//
//   One wave (32 lanes) per (16-query tile = blockIdx.x, b*h = blockIdx.y).
//   LDS: sS[16*16] (the masked scores A), sAcc[16*D] (output accumulator).
//   Per 16-key tile kt (a runtime scf.for up to the causal/ragged limit):
//     1. A = φ(Q) @ φ(K)ᵀ   — WMMA accumulate over D/16 head-dim chunks; the
//        feature map φ is applied on the loaded Q/K fragment elements.
//     2. mask (ragged key / causal) -> sS, MULTIPLICATIVELY (masked entries
//        become 0, the linear-attention analog of softmax's −∞).
//     3. O += A @ V         — WMMA accumulate over head-dim chunks. A is reread
//        from LDS in the A@V A-fragment layout (the same layout bridge flash
//        attention uses for P@V).
//   Final: O = sAcc (NO divide — linear attention is unnormalized).
//
// `head_dim` (D, a multiple of 16) is compile-time; query/key and value head
// dim are assumed equal. `dtype` is the f16/bf16 storage (f32 accumulate). The
// score WMMA (`tessera_rocm.wmma`) lowers through Stage J to the real
// `rocdl.wmma`. Validated against the canonical linear-attention reference
// `O = (φ(Q)φ(K)ᵀ ⊙ tril) @ V` on gfx1151.
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

void emitLinearAttnBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, int64_t D,
                        Type storeTy, StringRef featureMap, bool decay,
                        bool viaTile) {
  MLIRContext *ctx = b.getContext();
  int64_t DC = D / 16;
  Type f32 = b.getF32Type();
  auto fragTy = VectorType::get({16}, storeTy);
  auto accTy = VectorType::get({8}, f32);
  auto slt = arith::CmpIPredicate::slt;
  auto sge = arith::CmpIPredicate::sge;
  auto ne = arith::CmpIPredicate::ne;
  bool isRelu = (featureMap == "relu");
  bool isPoly2 = (featureMap == "polynomial_2");

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsT = [&](int64_t n, Type e) {
    return MemRefType::get({n}, e, MemRefLayoutAttrInterface(), ws);
  };
  // No softmax stats (sm/sl/scorr) — linear attention is unnormalized.
  Value sS = f.addWorkgroupAttribution(ldsT(16 * 16, f32), loc);
  Value sAcc = f.addWorkgroupAttribution(ldsT(16 * D, f32), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value Q = f.getArgument(0), Kk = f.getArgument(1), V = f.getArgument(2);
  Value O = f.getArgument(3);
  Value Sq = f.getArgument(4), Sk = f.getArgument(5), causal = f.getArgument(6);
  // Decay-masked variants (lightning_attention / retention): per-head log λ is
  // the trailing f32 arg; A[i,j] *= exp((i-j)·log_decay) over the causal band.
  Value logDecay;
  if (decay)
    logDecay = f.getArgument(7);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), c4 = ci(4), c15 = ci(15), c16 = ci(16),
        c32 = ci(32);
  Value cD = ci(D), c16D = ci(16 * D);
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));
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

  // Elementwise feature map φ on a loaded store-type element.
  auto phi = [&](Value v) -> Value {
    if (isRelu)
      return b.create<arith::MaximumFOp>(loc, v, storeZero);
    if (isPoly2)
      return b.create<arith::MulFOp>(loc, v, v); // x²
    return v; // identity
  };

  // zero sAcc: for i = tid; i < 16*D; i += 32.
  {
    auto lp = b.create<scf::ForOp>(loc, tid, c16D, c32);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    b.create<memref::StoreOp>(loc, zerof, sAcc,
                              ValueRange{lp.getInductionVar()});
  }
  b.create<gpu::BarrierOp>(loc);

  // nKV = (Sk+15)/16 ; lastKt = min(causal ? (q0+15)/16 : nKV-1, nKV-1).
  Value nKV = b.create<arith::DivUIOp>(loc, add(Sk, c15), c16);
  Value nKVm1 = b.create<arith::SubIOp>(loc, nKV, c1);
  Value isCausal = b.create<arith::CmpIOp>(loc, ne, causal, c0);
  Value ckt = b.create<arith::DivUIOp>(loc, add(q0, c15), c16);
  Value lastKt = b.create<arith::SelectOp>(loc, isCausal, ckt, nKVm1);
  Value over = b.create<arith::CmpIOp>(loc, slt, nKVm1, lastKt);
  lastKt = b.create<arith::SelectOp>(loc, over, nKVm1, lastKt);
  Value upper = add(lastKt, c1);

  auto buildFrag = [&](OpBuilder &bb, Location l,
                       function_ref<Value(int64_t)> elt) {
    Value fr = fragZero;
    for (int64_t i = 0; i < 16; ++i)
      fr = bb.create<vector::InsertOp>(l, elt(i), fr, ArrayRef<int64_t>{i});
    return fr;
  };
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

    // A = φ(Q) @ φ(K)ᵀ over D/16 chunks. Q/K read from global; φ applied per
    // loaded element. masked rows/cols loaded as 0.
    Value qrow_l15 = add(q0, l15);
    Value qrInb = b.create<arith::CmpIOp>(loc, slt, qrow_l15, Sq);
    Value cs = accZero;
    for (int64_t dc = 0; dc < DC; ++dc) {
      Value dc16 = ci(dc * 16);
      Value aBase = add(add(qbase, mul(qrow_l15, cD)), dc16);
      Value aSafe = b.create<arith::SelectOp>(loc, qrInb, aBase, c0);
      Value aFrag = buildFrag(b, loc, [&](int64_t i) {
        Value v = b.create<memref::LoadOp>(loc, Q, ValueRange{add(aSafe, ci(i))});
        Value pv = phi(v);
        return b.create<arith::SelectOp>(loc, qrInb, pv, storeZero);
      });
      Value bBase = add(add(kbase, mul(kr_l15, cD)), dc16);
      Value bSafe = b.create<arith::SelectOp>(loc, krInb, bBase, c0);
      Value bFrag = buildFrag(b, loc, [&](int64_t i) {
        Value v = b.create<memref::LoadOp>(loc, Kk, ValueRange{add(bSafe, ci(i))});
        Value pv = phi(v);
        return b.create<arith::SelectOp>(loc, krInb, pv, storeZero);
      });
      cs = wmma(b, loc, aFrag, bFrag, cs);
    }
    // mask -> sS[qi*16 + ki], qi = 2e+half, ki = l15. MULTIPLICATIVE: a masked
    // entry becomes 0 (vs −∞ for softmax). Mask = ragged key OR (causal future).
    for (int64_t e = 0; e < 8; ++e) {
      Value qi = add(ci(2 * e), half);
      Value gk = add(k0, l15);
      Value csv = b.create<vector::ExtractOp>(loc, cs, ArrayRef<int64_t>{e});
      Value gkOOB = b.create<arith::CmpIOp>(loc, sge, gk, Sk);
      Value qpos = add(q0, qi);
      // Decay scale: csv *= exp((qpos - gk)·log_decay) = λ^(i-j). Applied before
      // masking; masked entries are zeroed below regardless. (qpos-gk via index
      // sub is only used on the kept side; future keys are masked out.)
      if (decay) {
        Value age = b.create<arith::SubIOp>(loc, qpos, gk);
        Value agef = b.create<arith::IndexCastOp>(loc, b.getI64Type(), age);
        agef = b.create<arith::SIToFPOp>(loc, f32, agef);
        Value ex = b.create<math::ExpOp>(
            loc, b.create<arith::MulFOp>(loc, agef, logDecay));
        csv = b.create<arith::MulFOp>(loc, csv, ex);
      }
      Value cmask = b.create<arith::AndIOp>(
          loc, isCausal, b.create<arith::CmpIOp>(loc, slt, qpos, gk));
      Value masked = b.create<arith::OrIOp>(loc, gkOOB, cmask);
      Value v = b.create<arith::SelectOp>(loc, masked, zerof, csv);
      Value sIdx = add(mul(qi, c16), l15);
      b.create<memref::StoreOp>(loc, v, sS, ValueRange{sIdx});
    }
    b.create<gpu::BarrierOp>(loc);

    // O += A @ V over D/16 chunks. A reread from LDS (truncated to store type)
    // in the A-fragment layout; accumulate directly into sAcc (no rescale).
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
        b.create<memref::StoreOp>(loc, b.create<arith::AddFOp>(loc, cur, cpe),
                                  sAcc, ValueRange{idx});
      }
    }
    b.create<gpu::BarrierOp>(loc);
  }

  // O = sAcc (final, no divide), for i = tid; i < 16*D; i += 32.
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
    Value av = b.create<memref::LoadOp>(loc, sAcc, ValueRange{i});
    Value gidx = add(add(qbase, mul(gq, cD)), c);
    b.create<memref::StoreOp>(loc, av, O, ValueRange{gidx});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateWMMALinearAttnKernelPass
    : PassWrapper<GenerateWMMALinearAttnKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateWMMALinearAttnKernelPass)

  GenerateWMMALinearAttnKernelPass() = default;
  GenerateWMMALinearAttnKernelPass(const GenerateWMMALinearAttnKernelPass &other)
      : PassWrapper(other) {}

  Option<bool> viaTile{*this, "via-tile",
                       llvm::cl::desc("emit tile.mma (route through the wave/LDS "
                                      "pipeline) instead of tessera_rocm.wmma"),
                       llvm::cl::init(false)};

  StringRef getArgument() const final {
    return "generate-wmma-linear-attn-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.linear_attn directive into a fragment-"
           "materialized RDNA WMMA linear-attention forward gpu kernel "
           "(quadratic-parallel form, no softmax; compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, vector::VectorDialect,
                    arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect, tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.linear_attn")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto dAttr = op->getAttrOfType<IntegerAttr>("head_dim");
      if (!nameAttr || !dAttr) {
        op->emitError("tessera_rocm.linear_attn missing name/head_dim");
        return signalPassFailure();
      }
      int64_t D = dAttr.getInt();
      if (D <= 0 || D % 16 != 0) {
        op->emitError("generate-wmma-linear-attn-kernel: head_dim must be a "
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
          op->emitError("generate-wmma-linear-attn-kernel: dtype must be f16 or "
                        "bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      StringRef featureMap = "identity";
      if (auto a = op->getAttrOfType<StringAttr>("feature_map"))
        featureMap = a.getValue();
      if (featureMap != "identity" && featureMap != "relu" &&
          featureMap != "polynomial_2") {
        op->emitError("generate-wmma-linear-attn-kernel: feature_map must be "
                      "identity, relu, or polynomial_2 (got '")
            << featureMap << "')";
        return signalPassFailure();
      }
      bool decay = false;
      if (auto a = op->getAttrOfType<BoolAttr>("decay"))
        decay = a.getValue();

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto abv = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto of = MemRefType::get({ShapedType::kDynamic}, f32);
      // (Q, K, V : memref<?xstore>, O : memref<?xf32>, Sq, Sk, causal : index
      //  [, log_decay : f32])
      SmallVector<Type> argTys{abv, abv, abv, of, idxTy, idxTy, idxTy};
      if (decay)
        argTys.push_back(f32); // per-head log λ (RetNet/lightning decay)
      auto fnTy = b.getFunctionType(argTys, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitLinearAttnBody(body, loc, gpuFunc, D, storeTy, featureMap, decay,
                         viaTile);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateWMMALinearAttnKernelPass() {
  return std::make_unique<GenerateWMMALinearAttnKernelPass>();
}
