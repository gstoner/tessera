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
      auto fnTy = b.getFunctionType(
          {memF32, memF32, memF32, memF32, memF32, memF32, memF32, idxTy,
           f32, f32, f32, f32, f32, f32, f32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitOptBody(body, loc, gpuFunc, kindAttr.getValue());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMOptimizerKernelPass() {
  return std::make_unique<GenerateROCMOptimizerKernelPass>();
}
