//===- GenerateROCMDftKernel.cpp - direct DFT gpu kernel -----------------===//
//
// Expands `tessera_rocm.dft` into a flat one-thread-per-output-bin discrete
// Fourier transform over a batch of complex rows (interleaved re,im f32):
//
//   OUT[b,k] = Σ_j IN[b,j] · exp(sign · 2πi · k·j / n)
//
// sign = +1 for inverse (UNNORMALIZED — the runtime applies the plan's scale),
// −1 for forward. One thread per (row, k); the j-sum is an scf.for with
// math.cos/sin twiddles (→ ROCDL). This is the correctness foundation for the
// ROCm spectral lane — it handles ANY length (the radix-2 / Bluestein perf path
// is a follow-up; the SpectralPlan still records the chosen strategy). O(n²);
// the runtime uses it for fft/ifft/rfft/irfft. Validated vs np.fft on gfx1151.
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
static constexpr double kTwoPi = 6.283185307179586;

void emitDftBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, double sign) {
  Type f32 = b.getF32Type();
  Type idx = b.getIndexType();
  Type i64 = b.getI64Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value IN = f.getArgument(0), OUT = f.getArgument(1);
  Value B = f.getArgument(2), N = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value total = b.create<arith::MulIOp>(loc, B, N);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value brow = b.create<arith::DivUIOp>(loc, gid, N);
  Value k = b.create<arith::RemUIOp>(loc, gid, N);
  Value base = b.create<arith::MulIOp>(loc, brow, N);     // row offset (complex)
  auto toF32 = [&](Value v) {
    return b.create<arith::SIToFPOp>(loc, f32,
                                     b.create<arith::IndexCastOp>(loc, i64, v));
  };
  Value kf = toF32(k);
  Value nf = toF32(N);
  Value coef = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr((float)(sign * kTwoPi)));
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);

  // sum over j (accumulate re, im)
  auto lp = b.create<scf::ForOp>(loc, c0, N, c1, ValueRange{zero, zero});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value j = lp.getInductionVar();
    Value accr = lp.getRegionIterArgs()[0], acci = lp.getRegionIterArgs()[1];
    Value off = b.create<arith::MulIOp>(loc, b.create<arith::AddIOp>(loc, base, j),
                                        c2);              // 2*(base+j)
    Value inr = b.create<memref::LoadOp>(loc, IN, ValueRange{off});
    Value ini = b.create<memref::LoadOp>(
        loc, IN, ValueRange{b.create<arith::AddIOp>(loc, off, c1)});
    // ang = coef * k * j / n
    Value kj = b.create<arith::MulFOp>(loc, kf, toF32(j));
    Value ang = b.create<arith::DivFOp>(loc, b.create<arith::MulFOp>(loc, coef, kj),
                                        nf);
    Value c = b.create<math::CosOp>(loc, ang);
    Value s = b.create<math::SinOp>(loc, ang);
    // re += inr*c - ini*s ; im += inr*s + ini*c
    Value nr = b.create<arith::AddFOp>(
        loc, accr,
        b.create<arith::SubFOp>(loc, b.create<arith::MulFOp>(loc, inr, c),
                                b.create<arith::MulFOp>(loc, ini, s)));
    Value ni = b.create<arith::AddFOp>(
        loc, acci,
        b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, inr, s),
                                b.create<arith::MulFOp>(loc, ini, c)));
    b.create<scf::YieldOp>(loc, ValueRange{nr, ni});
  }
  Value ooff = b.create<arith::MulIOp>(loc, gid, c2);
  b.create<memref::StoreOp>(loc, lp.getResult(0), OUT, ValueRange{ooff});
  b.create<memref::StoreOp>(loc, lp.getResult(1), OUT,
                            ValueRange{b.create<arith::AddIOp>(loc, ooff, c1)});
  (void)idx;
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMDftKernelPass
    : PassWrapper<GenerateROCMDftKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMDftKernelPass)

  StringRef getArgument() const final { return "generate-rocm-dft-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.dft directive into a direct one-thread-per-bin "
           "DFT gpu kernel (interleaved complex f32)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.dft")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.dft missing name");
        return signalPassFailure();
      }
      double sign = -1.0;
      if (auto a = op->getAttrOfType<BoolAttr>("inverse"))
        sign = a.getValue() ? 1.0 : -1.0;
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType({memTy, memTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitDftBody(body, loc, gpuFunc, sign);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMDftKernelPass() {
  return std::make_unique<GenerateROCMDftKernelPass>();
}
