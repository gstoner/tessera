//===- GenerateROCMFpQuantKernel.cpp - low-precision float-grid quant -----===//
//
// Expands `tessera_rocm.fpquant` into a flat 1-operand elementwise kernel that
// snaps each value to a low-precision float grid (the format-agnostic
// mantissa-snap, the ROCm mirror of avx512_fpquant_f32):
//
//   ax      = min(|x|, max_normal)
//   e       = max(floor(log2(ax)), min_exp)      (flat subnormal grid)
//   ulp     = 2^(e - mantissa_bits)
//   rounded = min(roundeven(ax / ulp) * ulp, max_normal)
//   out     = NaN if x is NaN else sign(x) * (ax>0 ? rounded : 0)
//
// `max_normal`/`mantissa_bits`/`min_exp` are compile-time attrs (parameterizing
// fp8 e4m3/e5m2, fp6 e2m3/e3m2, fp4 e2m1). log2/floor/exp2/roundeven lower
// through math→ROCDL. The runtime applies the per-tensor / per-block scale.
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

#include <limits>

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitFpQuantBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                     double maxNormal, int64_t mantissaBits, int64_t minExp) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1), N = f.getArgument(2);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value rv = b.create<memref::LoadOp>(loc, X, ValueRange{gid});
  Value v = isF32 ? rv : b.create<arith::ExtFOp>(loc, f32, rv);
  auto cst = [&](double val) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(val))
        .getResult();
  };
  Value zero = cst(0.0), vmax = cst(maxNormal);
  Value nan = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO, v, v);
  Value ax = b.create<arith::MinimumFOp>(loc, b.create<math::AbsFOp>(loc, v),
                                         vmax);
  Value pos = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, ax, zero);
  Value e = b.create<arith::MaximumFOp>(
      loc, b.create<math::FloorOp>(loc, b.create<math::Log2Op>(loc, ax)),
      cst((double)minExp));
  Value ulp = b.create<math::Exp2Op>(
      loc, b.create<arith::SubFOp>(loc, e, cst((double)mantissaBits)));
  Value q = b.create<math::RoundEvenOp>(loc, b.create<arith::DivFOp>(loc, ax, ulp));
  Value rounded = b.create<arith::MinimumFOp>(
      loc, b.create<arith::MulFOp>(loc, q, ulp), vmax);
  rounded = b.create<arith::SelectOp>(loc, pos, rounded, zero);
  Value neg = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, v, zero);
  Value signed_ =
      b.create<arith::SelectOp>(loc, neg, b.create<arith::NegFOp>(loc, rounded),
                                rounded);
  Value y = b.create<arith::SelectOp>(loc, nan, cst(std::numeric_limits<double>::quiet_NaN()),
                                      signed_);
  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMFpQuantKernelPass
    : PassWrapper<GenerateROCMFpQuantKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMFpQuantKernelPass)

  StringRef getArgument() const final { return "generate-rocm-fpquant-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.fpquant directive into a flat 1-operand "
           "elementwise low-precision float-grid quantization gpu kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.fpquant")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.fpquant missing name");
        return signalPassFailure();
      }
      double maxNormal = 6.0;
      if (auto a = op->getAttrOfType<FloatAttr>("max_normal"))
        maxNormal = a.getValueAsDouble();
      int64_t mantissaBits = 1;
      if (auto a = op->getAttrOfType<IntegerAttr>("mantissa_bits"))
        mantissaBits = a.getInt();
      int64_t minExp = -126;
      if (auto a = op->getAttrOfType<IntegerAttr>("min_exp"))
        minExp = a.getInt();
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
          op->emitError("generate-rocm-fpquant-kernel: dtype must be f32, f16, "
                        "or bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto fnTy = b.getFunctionType({memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitFpQuantBody(body, loc, gpuFunc, storeTy, maxNormal, mantissaBits,
                      minExp);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMFpQuantKernelPass() {
  return std::make_unique<GenerateROCMFpQuantKernelPass>();
}
