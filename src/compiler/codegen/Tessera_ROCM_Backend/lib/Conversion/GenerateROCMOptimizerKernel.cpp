//===- GenerateROCMOptimizerKernel.cpp - fused optimizer-step gpu kernel --===//
//
// Expands `tessera_rocm.optimizer` into a fused per-parameter optimizer update,
// one thread per parameter element. The `kind` StrAttr selects the rule at
// codegen (one cached kernel per optimizer); the bias-correction scalars
// (1-β^t) are computed on the host and passed in:
//
//   sgd      : p -= lr·g
//   momentum : v = β1·v + g ; p -= lr·v
//   nesterov : v = β1·v + g ; p -= lr·(g + β1·v)   (look-ahead momentum)
//   adam     : m=β1·m+(1-β1)g ; v=β2·v+(1-β2)g² ; p -= lr·(m/b1c)/(√(v/b2c)+eps)
//   adamw    : p *= (1-lr·wd) ; then adam (decoupled decay)
//   lion     : u=β1·m+(1-β1)g ; m=β2·m+(1-β2)g ; p *= (1-lr·wd) ; p -= lr·sign(u)
//
// Buffers p/g/m/v in, p_out/m_out/v_out out. Scalars (lr,β1,β2,eps,wd,b1c,b2c)
// are f32 kernel args. √ via math→ROCDL. All f32. CPU analog:
// avx512_optimizer_f32.
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

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

// Emit the per-element update for `kind`. gid indexes the flat parameter array;
// p/g/m/v are loaded, the new param/state stored.
void emitOptBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, StringRef kind) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value P = f.getArgument(0), G = f.getArgument(1), M = f.getArgument(2);
  Value V = f.getArgument(3), POUT = f.getArgument(4), MOUT = f.getArgument(5);
  Value VOUT = f.getArgument(6);
  Value N = f.getArgument(7);
  Value lr = f.getArgument(8), b1 = f.getArgument(9), b2 = f.getArgument(10);
  Value eps = f.getArgument(11), wd = f.getArgument(12);
  Value b1c = f.getArgument(13), b2c = f.getArgument(14);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  auto ld = [&](Value buf) {
    return b.create<memref::LoadOp>(loc, buf, ValueRange{gid}).getResult();
  };
  Value pi = ld(P), gi = ld(G);
  auto st = [&](Value buf, Value val) {
    b.create<memref::StoreOp>(loc, val, buf, ValueRange{gid});
  };

  if (kind == "sgd") {
    // p - lr*g
    st(POUT, b.create<arith::SubFOp>(loc, pi,
                                     b.create<arith::MulFOp>(loc, lr, gi)));
  } else if (kind == "momentum") {
    Value vv = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, ld(V)),
                                       gi);
    st(VOUT, vv);
    st(POUT, b.create<arith::SubFOp>(loc, pi,
                                     b.create<arith::MulFOp>(loc, lr, vv)));
  } else if (kind == "nesterov") {
    // v = β1·v + g ; p -= lr·(g + β1·v)  (look-ahead momentum)
    Value vv = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, ld(V)),
                                       gi);
    st(VOUT, vv);
    Value upd = b.create<arith::AddFOp>(loc, gi,
                                        b.create<arith::MulFOp>(loc, b1, vv));
    st(POUT, b.create<arith::SubFOp>(loc, pi,
                                     b.create<arith::MulFOp>(loc, lr, upd)));
  } else if (kind == "lion") {
    Value mi = ld(M);
    Value om1 = b.create<arith::SubFOp>(loc, one, b1);
    Value om2 = b.create<arith::SubFOp>(loc, one, b2);
    Value u = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, mi),
                                      b.create<arith::MulFOp>(loc, om1, gi));
    st(MOUT, b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b2, mi),
                                     b.create<arith::MulFOp>(loc, om2, gi)));
    // sign(u) = (u>0) ? 1 : (u<0 ? -1 : 0)
    Value pos = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, u, zero);
    Value neg = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, u, zero);
    Value negone = b.create<arith::SubFOp>(loc, zero, one);
    Value sgn = b.create<arith::SelectOp>(
        loc, pos, one, b.create<arith::SelectOp>(loc, neg, negone, zero));
    // p *= (1-lr*wd)
    Value pd = b.create<arith::MulFOp>(
        loc, pi, b.create<arith::SubFOp>(
                     loc, one, b.create<arith::MulFOp>(loc, lr, wd)));
    st(POUT, b.create<arith::SubFOp>(loc, pd,
                                     b.create<arith::MulFOp>(loc, lr, sgn)));
  } else {  // adam / adamw
    Value mi = ld(M), vi = ld(V);
    Value om1 = b.create<arith::SubFOp>(loc, one, b1);
    Value om2 = b.create<arith::SubFOp>(loc, one, b2);
    Value mm = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, b1, mi),
                                       b.create<arith::MulFOp>(loc, om1, gi));
    Value vv = b.create<arith::AddFOp>(
        loc, b.create<arith::MulFOp>(loc, b2, vi),
        b.create<arith::MulFOp>(loc, om2, b.create<arith::MulFOp>(loc, gi, gi)));
    st(MOUT, mm);
    st(VOUT, vv);
    Value pbase = pi;
    if (kind == "adamw")
      pbase = b.create<arith::MulFOp>(
          loc, pi, b.create<arith::SubFOp>(
                       loc, one, b.create<arith::MulFOp>(loc, lr, wd)));
    Value denom = b.create<arith::AddFOp>(
        loc, b.create<math::SqrtOp>(loc, b.create<arith::DivFOp>(loc, vv, b2c)),
        eps);
    Value upd = b.create<arith::DivFOp>(
        loc, b.create<arith::DivFOp>(loc, mm, b1c), denom);
    st(POUT, b.create<arith::SubFOp>(loc, pbase,
                                     b.create<arith::MulFOp>(loc, lr, upd)));
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitSgdBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value dy = f.getArgument(0), dParam = f.getArgument(1);
  Value dGrad = f.getArgument(2), n = f.getArgument(3);
  Value lr = f.getArgument(4);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());
  Value incoming = b.create<memref::LoadOp>(loc, dy, ValueRange{gid});
  b.create<memref::StoreOp>(loc, incoming, dParam, ValueRange{gid});
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  Value scaled = b.create<arith::MulFOp>(loc, lr, incoming);
  b.create<memref::StoreOp>(
      loc, b.create<arith::SubFOp>(loc, zero, scaled), dGrad,
      ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitMomentumBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                              bool nesterov) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value dParamOut = f.getArgument(0), dVelocityOut = f.getArgument(1);
  Value dParam = f.getArgument(2), dGrad = f.getArgument(3);
  Value dVelocity = f.getArgument(4), n = f.getArgument(5);
  Value lr = f.getArgument(6), mu = f.getArgument(7);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());
  Value dp =
      b.create<memref::LoadOp>(loc, dParamOut, ValueRange{gid});
  Value dv =
      b.create<memref::LoadOp>(loc, dVelocityOut, ValueRange{gid});
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  Value one = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(1.0f));
  Value fromParam = b.create<arith::MulFOp>(
      loc, b.create<arith::SubFOp>(loc, zero, lr), dp);
  Value gradFactor =
      nesterov ? b.create<arith::AddFOp>(loc, one, mu).getResult() : one;
  Value dg = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, gradFactor, fromParam), dv);
  Value velocityFactor = nesterov ? mu : one;
  Value velocityBase = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, velocityFactor, fromParam), dv);
  Value oldVelocityGrad =
      b.create<arith::MulFOp>(loc, mu, velocityBase);
  b.create<memref::StoreOp>(loc, dp, dParam, ValueRange{gid});
  b.create<memref::StoreOp>(loc, dg, dGrad, ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, oldVelocityGrad, dVelocity, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitLionBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  // Lion treats sign(beta1*m + (1-beta1)*g) as stop-gradient.  Consequently
  // the parameter output contributes only through decoupled weight decay,
  // while the carried moment output contributes the affine beta2 update.
  b.setInsertionPointToStart(&f.getBody().front());
  Value dParamOut = f.getArgument(0), dMomentOut = f.getArgument(1);
  Value dParam = f.getArgument(2), dGrad = f.getArgument(3);
  Value dMoment = f.getArgument(4), n = f.getArgument(5);
  Value lr = f.getArgument(6), beta2 = f.getArgument(7);
  Value weightDecay = f.getArgument(8);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());
  Type f32 = b.getF32Type();
  Value one = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f));
  Value dpOut =
      b.create<memref::LoadOp>(loc, dParamOut, ValueRange{gid});
  Value dmOut =
      b.create<memref::LoadOp>(loc, dMomentOut, ValueRange{gid});
  Value parameterFactor = b.create<arith::SubFOp>(
      loc, one, b.create<arith::MulFOp>(loc, lr, weightDecay));
  Value gradientFactor = b.create<arith::SubFOp>(loc, one, beta2);
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, parameterFactor, dpOut), dParam,
      ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, gradientFactor, dmOut), dGrad,
      ValueRange{gid});
  b.create<memref::StoreOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, dmOut), dMoment,
      ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

void emitAdamBackwardBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                          bool adamw) {
  b.setInsertionPointToStart(&f.getBody().front());
  Value grad = f.getArgument(1), moment1 = f.getArgument(2);
  Value moment2 = f.getArgument(3), dParamOut = f.getArgument(4);
  Value dMoment1Out = f.getArgument(5), dMoment2Out = f.getArgument(6);
  Value dParam = f.getArgument(7), dGrad = f.getArgument(8);
  Value dMoment1 = f.getArgument(9), dMoment2 = f.getArgument(10);
  Value n = f.getArgument(11), lr = f.getArgument(12);
  Value beta1 = f.getArgument(13), beta2 = f.getArgument(14);
  Value eps = f.getArgument(15), weightDecay = f.getArgument(16);
  Value beta1Correction = f.getArgument(17);
  Value beta2Correction = f.getArgument(18);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, block), tid);
  Value inBounds =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gid, n);
  auto guard = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Type f32 = b.getF32Type();
  auto constant = [&](float value) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(value))
        .getResult();
  };
  auto load = [&](Value buffer) {
    return b.create<memref::LoadOp>(loc, buffer, ValueRange{gid}).getResult();
  };
  auto store = [&](Value value, Value buffer) {
    b.create<memref::StoreOp>(loc, value, buffer, ValueRange{gid});
  };
  Value zero = constant(0.0f), one = constant(1.0f);
  Value g = load(grad), m = load(moment1), v = load(moment2);
  Value dpOut = load(dParamOut), dmOut = load(dMoment1Out);
  Value dvOut = load(dMoment2Out);
  Value oneMinusBeta1 = b.create<arith::SubFOp>(loc, one, beta1);
  Value oneMinusBeta2 = b.create<arith::SubFOp>(loc, one, beta2);
  Value mNew = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, beta1, m),
      b.create<arith::MulFOp>(loc, oneMinusBeta1, g));
  Value vNew = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, beta2, v),
      b.create<arith::MulFOp>(
          loc, oneMinusBeta2, b.create<arith::MulFOp>(loc, g, g)));
  Value normalizedV =
      b.create<arith::DivFOp>(loc, vNew, beta2Correction);
  Value root = b.create<math::SqrtOp>(loc, normalizedV);
  Value denom = b.create<arith::AddFOp>(loc, root, eps);
  Value negativeLr =
      b.create<arith::SubFOp>(loc, zero, lr);
  Value dMFromParam = b.create<arith::DivFOp>(
      loc,
      b.create<arith::MulFOp>(
          loc, dpOut,
          b.create<arith::DivFOp>(loc, negativeLr, beta1Correction)),
      denom);
  Value dMNew = b.create<arith::AddFOp>(loc, dmOut, dMFromParam);
  Value numerator =
      b.create<arith::DivFOp>(loc, mNew, beta1Correction);
  Value positive = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, normalizedV, zero);
  Value dRoot = b.create<arith::SelectOp>(
      loc, positive,
      b.create<arith::DivFOp>(
          loc, constant(0.5f),
          b.create<arith::MulFOp>(loc, beta2Correction, root)),
      zero);
  Value denomSquared = b.create<arith::MulFOp>(loc, denom, denom);
  Value dVFromParam = b.create<arith::MulFOp>(
      loc, dpOut,
      b.create<arith::MulFOp>(
          loc, lr,
          b.create<arith::DivFOp>(
              loc, b.create<arith::MulFOp>(loc, numerator, dRoot),
              denomSquared)));
  Value dVNew = b.create<arith::AddFOp>(loc, dvOut, dVFromParam);
  Value paramFactor = one;
  if (adamw)
    paramFactor = b.create<arith::SubFOp>(
        loc, one, b.create<arith::MulFOp>(loc, lr, weightDecay));
  Value gradFromMoment1 =
      b.create<arith::MulFOp>(loc, oneMinusBeta1, dMNew);
  Value gradFromMoment2 = b.create<arith::MulFOp>(
      loc, constant(2.0f),
      b.create<arith::MulFOp>(
          loc, oneMinusBeta2, b.create<arith::MulFOp>(loc, g, dVNew)));
  store(b.create<arith::MulFOp>(loc, dpOut, paramFactor), dParam);
  store(b.create<arith::AddFOp>(loc, gradFromMoment1, gradFromMoment2),
        dGrad);
  store(b.create<arith::MulFOp>(loc, beta1, dMNew), dMoment1);
  store(b.create<arith::MulFOp>(loc, beta2, dVNew), dMoment2);
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMOptimizerKernelPass
    : PassWrapper<GenerateROCMOptimizerKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMOptimizerKernelPass)

  StringRef getArgument() const final { return "generate-rocm-optimizer-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.optimizer directive into a fused per-parameter "
           "optimizer-step gpu kernel (kind StrAttr selects the rule)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.optimizer")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto kindAttr = op->getAttrOfType<StringAttr>("kind");
      if (!nameAttr || !kindAttr) {
        op->emitError("tessera_rocm.optimizer missing name/kind");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      bool backward = false;
      if (auto attr = op->getAttrOfType<BoolAttr>("backward"))
        backward = attr.getValue();
      bool momentumBackward =
          backward && (kindAttr.getValue() == "momentum" ||
                       kindAttr.getValue() == "nesterov");
      bool adamBackward =
          backward && (kindAttr.getValue() == "adam" ||
                       kindAttr.getValue() == "adamw");
      bool lionBackward = backward && kindAttr.getValue() == "lion";
      auto fnTy = backward
                      ? (adamBackward
                             ? b.getFunctionType(
                                   {memF32, memF32, memF32, memF32, memF32,
                                    memF32, memF32, memF32, memF32, memF32,
                                    memF32, idxTy, f32, f32, f32, f32, f32,
                                   f32, f32},
                                   {})
                             : lionBackward
                             ? b.getFunctionType(
                                   {memF32, memF32, memF32, memF32, memF32,
                                    idxTy, f32, f32, f32},
                                   {})
                             : momentumBackward
                             ? b.getFunctionType(
                                   {memF32, memF32, memF32, memF32, memF32,
                                    idxTy, f32, f32},
                                   {})
                             : b.getFunctionType(
                                   {memF32, memF32, memF32, idxTy, f32}, {}))
                      : b.getFunctionType(
                            {memF32, memF32, memF32, memF32, memF32, memF32,
                             memF32, idxTy, f32, f32, f32, f32, f32, f32,
                             f32},
                            {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (backward) {
        if (adamBackward)
          emitAdamBackwardBody(body, loc, gpuFunc,
                               kindAttr.getValue() == "adamw");
        else if (lionBackward)
          emitLionBackwardBody(body, loc, gpuFunc);
        else if (momentumBackward)
          emitMomentumBackwardBody(body, loc, gpuFunc,
                                   kindAttr.getValue() == "nesterov");
        else if (kindAttr.getValue() == "sgd")
          emitSgdBackwardBody(body, loc, gpuFunc);
        else {
          op->emitError(
              "optimizer backward supports sgd, momentum, nesterov, adam, "
              "adamw, lion");
          return signalPassFailure();
        }
      } else {
        emitOptBody(body, loc, gpuFunc, kindAttr.getValue());
      }
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMOptimizerKernelPass() {
  return std::make_unique<GenerateROCMOptimizerKernelPass>();
}
