//===- GenerateROCMDeltaNetKernel.cpp - gated/delta linear-attention scan -===//
//
// Expands a `tessera_rocm.deltanet` directive into a causal sequential-scan gpu
// kernel for the gated/delta linear-attention recurrence (the compiled lane for
// gated_deltanet / kimi_delta_attention / modified_delta_attention). One
// workgroup per (b,h); one thread per value-column e (blockDim = D_v). Thread e
// owns the state column Ŝ[:,e] (length D_qk) in LDS, so the per-step matvecs are
// independent per-thread d-loops with no cross-thread state hazard — the only
// barriers are the cooperative k/q load and the modified-delta ‖target‖
// reduction.
//
// Per timestep t (matching `_delta_attention_impl`, causal path):
//   target = V[t]
//   if erase:   target -= α_t · (Ŝᵀ k_t)         // read OLD state
//   if decay:   Ŝ *= α_t
//   delta = outer(k_t, target)  [/ (1 + ‖k_t‖·‖target‖) if modified]
//   Ŝ += β_t · delta
//   O[t] = Q_t @ Ŝ                                // read NEW state
//   if gate: O[t] *= sigmoid(gate[t])
//
// D_qk/D_v are compile-time (static loop bounds + LDS sizing); the flags
// (erase / modified / has_gate / has_beta / has_decay) are emitted as
// straight-line code. Compute is f32; storage is f16/bf16/f32. Validated vs the
// numpy reference on gfx1151. State starts at zero (state=None); no state out.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct DeltaNetFlags {
  bool erase = false;
  bool modified = false;
  bool hasGate = false;
  bool hasBeta = false;
  bool hasDecay = false;
};

void emitDeltaNetBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                      int64_t Dqk, int64_t Dv, DeltaNetFlags fl) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i64 = b.getIntegerType(64);
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  auto eqp = arith::CmpIPredicate::eq;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto lds = [&](int64_t n) {
    return f.addWorkgroupAttribution(
        MemRefType::get({n}, f32, MemRefLayoutAttrInterface(), ws), loc);
  };
  Value state = lds(Dv * Dqk);   // Ŝ[e*Dqk + d] — thread e owns column e
  Value kSh = lds(Dqk);          // current k_t
  Value qSh = lds(Dqk);          // current q_t
  Value red = lds(Dv);           // ‖target‖ reduction scratch (modified only)

  b.setInsertionPointToStart(&f.getBody().front());
  // (Q, K : memref<?xstore>[BH*S*Dqk], V, O : [BH*S*Dv],
  //  gate : [BH*S*Dv], beta, decay : f32[BH*S], S : index)
  Value Q = f.getArgument(0), K = f.getArgument(1), V = f.getArgument(2),
        O = f.getArgument(3), gate = f.getArgument(4), beta = f.getArgument(5),
        decay = f.getArgument(6), S = f.getArgument(7);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  Value c0 = ci(0), c1 = ci(1), cDqk = ci(Dqk), cDv = ci(Dv);
  Value zero = cf(0.0), one = cf(1.0);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value e = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);  // value-col, tid

  // base offsets: qk_base = bid*S*Dqk ; v_base = bid*S*Dv ; sc_base = bid*S
  Value SDqk = b.create<arith::MulIOp>(loc, S, cDqk);
  Value SDv = b.create<arith::MulIOp>(loc, S, cDv);
  Value qkBase = b.create<arith::MulIOp>(loc, bid, SDqk);
  Value vBase = b.create<arith::MulIOp>(loc, bid, SDv);
  Value scBase = b.create<arith::MulIOp>(loc, bid, S);

  auto extF = [&](Value mr, Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, mr, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  auto stCol = [&](Value d) {  // state index for this thread's column e
    return b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, e, cDqk), d);
  };

  // init Ŝ[e,:] = 0
  {
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    b.create<memref::StoreOp>(loc, zero, state,
                              ValueRange{stCol(lp.getInductionVar())});
  }
  b.create<gpu::BarrierOp>(loc);

  // sequential scan over t in [0, S)
  auto tloop = b.create<scf::ForOp>(loc, c0, S, c1);
  b.setInsertionPointToStart(tloop.getBody());
  Value t = tloop.getInductionVar();
  Value tDqk = b.create<arith::MulIOp>(loc, t, cDqk);
  Value tDv = b.create<arith::MulIOp>(loc, t, cDv);
  Value qkRow = b.create<arith::AddIOp>(loc, qkBase, tDqk);  // Q/K row start
  Value vRow = b.create<arith::AddIOp>(loc, vBase, tDv);     // V/O row start
  Value scIdx = b.create<arith::AddIOp>(loc, scBase, t);     // beta/decay index

  // cooperative load k_sh/q_sh (length Dqk), strided by blockDim=Dv
  {
    auto lp = b.create<scf::ForOp>(loc, e, cDqk, cDv);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value d = lp.getInductionVar();
    Value gidx = b.create<arith::AddIOp>(loc, qkRow, d);
    b.create<memref::StoreOp>(loc, extF(K, gidx), kSh, ValueRange{d});
    b.create<memref::StoreOp>(loc, extF(Q, gidx), qSh, ValueRange{d});
  }
  b.create<gpu::BarrierOp>(loc);

  // ---- thread e (value column) ----
  Value alpha = fl.hasDecay
                    ? b.create<memref::LoadOp>(loc, decay, ValueRange{scIdx})
                    : one;
  Value betaW = fl.hasBeta
                    ? b.create<memref::LoadOp>(loc, beta, ValueRange{scIdx})
                    : one;
  Value vIdx = b.create<arith::AddIOp>(loc, vRow, e);
  Value targetE = extF(V, vIdx);

  // d-loop reducer over this thread's state column: acc += f(d)
  auto colReduce = [&](function_ref<Value(Value /*d*/)> term) -> Value {
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1, ValueRange{zero});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value acc = lp.getRegionIterArgs()[0];
    Value v = term(lp.getInductionVar());
    b.create<scf::YieldOp>(loc,
                           ValueRange{b.create<arith::AddFOp>(loc, acc, v)});
    return lp.getResult(0);
  };

  if (fl.erase) {
    // v̂_e = Σ_d k_sh[d]·Ŝ[e,d]  (OLD state); target -= α·v̂
    Value vhat = colReduce([&](Value d) {
      Value kv = b.create<memref::LoadOp>(loc, kSh, ValueRange{d});
      Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{stCol(d)});
      return b.create<arith::MulFOp>(loc, kv, sv).getResult();
    });
    targetE = b.create<arith::SubFOp>(
        loc, targetE, b.create<arith::MulFOp>(loc, alpha, vhat));
  }

  if (fl.hasDecay) {  // Ŝ[e,:] *= α
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value idx = stCol(lp.getInductionVar());
    Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{idx});
    b.create<memref::StoreOp>(loc, b.create<arith::MulFOp>(loc, sv, alpha),
                              state, ValueRange{idx});
  }

  Value scale = one;
  if (fl.modified) {
    // scale = 1 / (1 + ‖k‖·‖target‖);  ‖delta‖_F = ‖k‖₂·‖target‖₂
    Value knorm2 = colReduce([&](Value d) {  // reuse d-loop shape over Dqk
      Value kv = b.create<memref::LoadOp>(loc, kSh, ValueRange{d});
      return b.create<arith::MulFOp>(loc, kv, kv).getResult();
    });
    Value knorm = b.create<math::SqrtOp>(loc, knorm2);
    // ‖target‖: cross-thread reduction over the Dv columns via LDS.
    b.create<memref::StoreOp>(
        loc, b.create<arith::MulFOp>(loc, targetE, targetE), red,
        ValueRange{e});
    b.create<gpu::BarrierOp>(loc);
    Value isT0 = b.create<arith::CmpIOp>(loc, eqp, e, c0);
    auto t0if = b.create<scf::IfOp>(loc, isT0, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(t0if.thenBlock());
      auto lp = b.create<scf::ForOp>(loc, c0, cDv, c1, ValueRange{zero});
      b.setInsertionPointToStart(lp.getBody());
      Value acc = lp.getRegionIterArgs()[0];
      Value rv = b.create<memref::LoadOp>(loc, red,
                                          ValueRange{lp.getInductionVar()});
      b.create<scf::YieldOp>(loc,
                             ValueRange{b.create<arith::AddFOp>(loc, acc, rv)});
      b.setInsertionPointAfter(lp);
      b.create<memref::StoreOp>(loc, lp.getResult(0), red, ValueRange{c0});
    }
    b.create<gpu::BarrierOp>(loc);
    Value tnorm =
        b.create<math::SqrtOp>(loc, b.create<memref::LoadOp>(loc, red,
                                                             ValueRange{c0}));
    Value denom = b.create<arith::AddFOp>(
        loc, one, b.create<arith::MulFOp>(loc, knorm, tnorm));
    scale = b.create<arith::DivFOp>(loc, one, denom);
    b.create<gpu::BarrierOp>(loc);  // all read red[0] before next-t red reuse
  }

  // Ŝ[e,d] += β · k[d] · target · scale
  Value bt = b.create<arith::MulFOp>(
      loc, betaW, b.create<arith::MulFOp>(loc, targetE, scale));
  {
    auto lp = b.create<scf::ForOp>(loc, c0, cDqk, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value d = lp.getInductionVar();
    Value idx = stCol(d);
    Value kv = b.create<memref::LoadOp>(loc, kSh, ValueRange{d});
    Value add = b.create<arith::MulFOp>(loc, bt, kv);
    Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{idx});
    b.create<memref::StoreOp>(loc, b.create<arith::AddFOp>(loc, sv, add), state,
                              ValueRange{idx});
  }

  // O_e = Σ_d q[d]·Ŝ[e,d]  (NEW state)
  Value oE = colReduce([&](Value d) {
    Value qv = b.create<memref::LoadOp>(loc, qSh, ValueRange{d});
    Value sv = b.create<memref::LoadOp>(loc, state, ValueRange{stCol(d)});
    return b.create<arith::MulFOp>(loc, qv, sv).getResult();
  });
  if (fl.hasGate) {  // O *= sigmoid(gate)
    Value g = extF(gate, vIdx);
    Value sig = b.create<arith::DivFOp>(
        loc, one,
        b.create<arith::AddFOp>(
            loc, one, b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, g))));
    oE = b.create<arith::MulFOp>(loc, oE, sig);
  }
  Value sv = isF32 ? oE : b.create<arith::TruncFOp>(loc, storeTy, oE);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{vIdx});

  b.create<gpu::BarrierOp>(loc);  // before next-t cooperative load reuses k/q
  // (end t-loop body)

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMDeltaNetKernelPass
    : PassWrapper<GenerateROCMDeltaNetKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMDeltaNetKernelPass)

  StringRef getArgument() const final { return "generate-rocm-deltanet-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.deltanet directive into a causal "
           "sequential-scan gated/delta linear-attention gpu kernel "
           "(compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.deltanet")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.deltanet missing name");
        return signalPassFailure();
      }
      auto dimAttr = [&](StringRef n) -> int64_t {
        if (auto a = op->getAttrOfType<IntegerAttr>(n)) return a.getInt();
        return 0;
      };
      int64_t Dqk = dimAttr("d_qk"), Dv = dimAttr("d_v");
      if (Dqk <= 0 || Dv <= 0) {
        op->emitError("generate-rocm-deltanet-kernel: d_qk and d_v must be "
                      "positive");
        return signalPassFailure();
      }
      auto flag = [&](StringRef n) -> bool {
        if (auto a = op->getAttrOfType<BoolAttr>(n)) return a.getValue();
        return false;
      };
      DeltaNetFlags fl;
      fl.erase = flag("erase");
      fl.modified = flag("modified");
      fl.hasGate = flag("has_gate");
      fl.hasBeta = flag("has_beta");
      fl.hasDecay = flag("has_decay");

      Type storeTy = b_storeType(op);
      if (!storeTy) return signalPassFailure();

      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto store = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto f32mr = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      // (Q,K,V,O,gate : store, beta,decay : f32, S : index)
      auto fnTy = b.getFunctionType(
          {store, store, store, store, store, f32mr, f32mr, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitDeltaNetBody(body, loc, gpuFunc, storeTy, Dqk, Dv, fl);
      op->erase();
    }
  }

  // dtype attr -> storage type (f32/f16/bf16); emits an error + returns null.
  Type b_storeType(Operation *op) {
    OpBuilder b(op->getContext());
    if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
      StringRef dt = a.getValue();
      if (dt == "f16" || dt == "float16") return b.getF16Type();
      if (dt == "bf16" || dt == "bfloat16") return b.getBF16Type();
      if (dt != "f32" && dt != "float32") {
        op->emitError("generate-rocm-deltanet-kernel: dtype must be f32, f16, "
                      "or bf16 (got '")
            << dt << "')";
        return nullptr;
      }
    }
    return b.getF32Type();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMDeltaNetKernelPass() {
  return std::make_unique<GenerateROCMDeltaNetKernelPass>();
}
