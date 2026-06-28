//===- GenerateROCMPointwiseLossKernel.cpp - pointwise regression loss ----===//
//
// Expands a `tessera_rocm.pointwise_loss` directive into a flat 2-operand
// elementwise gpu kernel computing the PER-ELEMENT S11 regression loss over N
// elements (one thread per element). The runtime applies the reduction
// (none/mean/sum) via the rocm reduce lane.
//
//   kind 0 = mse        (p−t)²
//   kind 1 = mae        |p−t|
//   kind 2 = huber(δ)   a=|p−t|; a≤δ ? ½a² : δ(a−½δ)
//   kind 3 = smooth_l1(β) a=|p−t|; a<β ? ½a²/β : a−½β
//   kind 4 = log_cosh   e + log1p(exp(−2e)) − log2
//
// `kind` and `param` (δ/β) are compile-time attrs (no per-thread branch).
// Computes in f32 regardless of storage dtype; exp/log1p lower through
// convert-math-to-rocdl. The CPU analog is avx512_loss_f32.cpp. Validated vs the
// tessera.losses reference on gfx1151.
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

static constexpr int64_t BD = 256;

void emitLossBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                  int64_t kind, double param) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), O = f.getArgument(2),
        N = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value rp = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value rt = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
  Value p = isF32 ? rp : b.create<arith::ExtFOp>(loc, f32, rp);
  Value t = isF32 ? rt : b.create<arith::ExtFOp>(loc, f32, rt);
  auto cst = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v))
        .getResult();
  };
  Value e = b.create<arith::SubFOp>(loc, p, t);
  Value ax = b.create<math::AbsFOp>(loc, e);
  Value half = cst(0.5);
  Value y;
  switch (kind) {
  case 0:  // mse
    y = b.create<arith::MulFOp>(loc, e, e);
    break;
  case 1:  // mae
    y = ax;
    break;
  case 2: {  // huber(delta)
    Value d = cst(param);
    Value le = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, ax, d);
    Value q = b.create<arith::MulFOp>(loc, half,
                                      b.create<arith::MulFOp>(loc, ax, ax));
    Value lin = b.create<arith::MulFOp>(
        loc, d, b.create<arith::SubFOp>(loc, ax,
                                        b.create<arith::MulFOp>(loc, half, d)));
    y = b.create<arith::SelectOp>(loc, le, q, lin);
    break;
  }
  case 3: {  // smooth_l1(beta)
    Value beta = cst(param);
    Value lt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, ax, beta);
    Value q = b.create<arith::DivFOp>(
        loc, b.create<arith::MulFOp>(loc, half,
                                     b.create<arith::MulFOp>(loc, ax, ax)),
        beta);
    Value lin = b.create<arith::SubFOp>(
        loc, ax, b.create<arith::MulFOp>(loc, half, beta));
    y = b.create<arith::SelectOp>(loc, lt, q, lin);
    break;
  }
  default: {  // log_cosh: e + log1p(exp(-2e)) - log2
    Value two = cst(2.0);
    Value nege2 = b.create<arith::NegFOp>(loc, b.create<arith::MulFOp>(loc, two, e));
    Value l = b.create<math::Log1pOp>(loc, b.create<math::ExpOp>(loc, nege2));
    y = b.create<arith::SubFOp>(loc, b.create<arith::AddFOp>(loc, e, l),
                                cst(0.6931471805599453));
    break;
  }
  }

  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMPointwiseLossKernelPass
    : PassWrapper<GenerateROCMPointwiseLossKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMPointwiseLossKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-pointwise-loss-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.pointwise_loss directive into a flat "
           "2-operand elementwise per-element regression-loss gpu kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.pointwise_loss")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.pointwise_loss missing name");
        return signalPassFailure();
      }
      int64_t kind = 0;
      if (auto k = op->getAttrOfType<IntegerAttr>("kind"))
        kind = k.getInt();
      if (kind < 0 || kind > 4) {
        op->emitError("generate-rocm-pointwise-loss-kernel: kind must be 0..4");
        return signalPassFailure();
      }
      double param = 1.0;
      if (auto pp = op->getAttrOfType<FloatAttr>("param"))
        param = pp.getValueAsDouble();

      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16") storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16") storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-pointwise-loss-kernel: dtype must be "
                        "f32, f16, or bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitLossBody(body, loc, gpuFunc, storeTy, kind, param);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMPointwiseLossKernelPass() {
  return std::make_unique<GenerateROCMPointwiseLossKernelPass>();
}
