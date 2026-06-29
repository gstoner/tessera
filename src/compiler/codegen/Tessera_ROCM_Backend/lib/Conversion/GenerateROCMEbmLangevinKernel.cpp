//===- GenerateROCMEbmLangevinKernel.cpp - Langevin step + Philox noise --===//
//
// Expands a `tessera_rocm.ebm_langevin` directive into a Langevin-step gpu
// kernel that DRAWS its own Gaussian noise from counter-based Philox-4x32-10
// (the P6 RNG generator) — the sampling half of the P7 EBM follow-up. One
// thread per element `i`:
//
//   counter = (c0 + i, c1, c2, c3);  philox-4x32-10 under key (k0, k1)
//   u0 = (out0 + 0.5)·2^-32;  u1 = (out1 + 0.5)·2^-32
//   z  = sqrt(-2 ln u0)·cos(2π u1)                  (Box-Muller, first lobe)
//   out[i] = y[i] - eta·grad[i] + noise_scale·z
//
// Mirrors `tessera.ebm.langevin_step_philox`: the counter layout, the `+0.5`
// uniform map, and the Box-Muller all match the numpy reference. The Philox
// round is the same one `generate-rocm-philox-kernel` emits. CPU analog:
// avx512_ebm_langevin_f32. Args: (y : memref<?xf32>, grad : memref<?xf32>,
// n : index, eta f32, noise_scale f32, k0 i32, k1 i32, c0 i32, c1 i32, c2 i32,
// c3 i32, out : memref<?xf32>).
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

void emitLangevinBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type i32 = b.getIntegerType(32);
  Type i64 = b.getIntegerType(64);
  Type f32 = b.getF32Type();
  Type idxTy = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value Y = f.getArgument(0), G = f.getArgument(1), N = f.getArgument(2);
  Value eta = f.getArgument(3), noiseScale = f.getArgument(4);
  Value k0i = f.getArgument(5), k1i = f.getArgument(6);
  Value c0i = f.getArgument(7), c1i = f.getArgument(8);
  Value c2i = f.getArgument(9), c3i = f.getArgument(10);
  Value OUT = f.getArgument(11);

  auto ci32 = [&](uint32_t v) {
    return b.create<arith::ConstantOp>(
        loc, i32, b.getIntegerAttr(i32, static_cast<int32_t>(v)));
  };
  Value M0 = ci32(0xD2511F53u), M1 = ci32(0xCD9E8D57u);
  Value W0 = ci32(0x9E3779B9u), W1 = ci32(0xBB67AE85u);
  Value c32 = b.create<arith::ConstantOp>(loc, i64, b.getIntegerAttr(i64, 32));

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  // counter (c0 + i, c1, c2, c3)
  Value iI32 = b.create<arith::IndexCastOp>(loc, i32, gid);
  Value c0 = b.create<arith::AddIOp>(loc, c0i, iI32);
  Value c1 = c1i, c2 = c2i, c3 = c3i;
  Value k0 = k0i, k1 = k1i;

  for (int r = 0; r < 10; ++r) {
    if (r > 0) {
      k0 = b.create<arith::AddIOp>(loc, k0, W0);
      k1 = b.create<arith::AddIOp>(loc, k1, W1);
    }
    Value p0 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c0),
        b.create<arith::ExtUIOp>(loc, i64, M0));
    Value p1 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c2),
        b.create<arith::ExtUIOp>(loc, i64, M1));
    Value hi0 = b.create<arith::TruncIOp>(
        loc, i32, b.create<arith::ShRUIOp>(loc, p0, c32));
    Value lo0 = b.create<arith::TruncIOp>(loc, i32, p0);
    Value hi1 = b.create<arith::TruncIOp>(
        loc, i32, b.create<arith::ShRUIOp>(loc, p1, c32));
    Value lo1 = b.create<arith::TruncIOp>(loc, i32, p1);
    Value n0 = b.create<arith::XOrIOp>(
        loc, b.create<arith::XOrIOp>(loc, hi1, c1), k0);
    Value n2 = b.create<arith::XOrIOp>(
        loc, b.create<arith::XOrIOp>(loc, hi0, c3), k1);
    c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
  }

  // Box-Muller: u = (uitofp(c) + 0.5) * 2^-32; z = sqrt(-2 ln u0)·cos(2π u1).
  Value half = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.5f));
  Value inv = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f / 4294967296.0f));
  auto uniform = [&](Value cw) -> Value {
    Value fp = b.create<arith::AddFOp>(
        loc, b.create<arith::UIToFPOp>(loc, f32, cw), half);
    return b.create<arith::MulFOp>(loc, fp, inv);
  };
  Value u0 = uniform(c0), u1 = uniform(c1);
  Value negTwo = b.create<arith::ConstantOp>(loc, f32,
                                             b.getF32FloatAttr(-2.0f));
  Value twoPi = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(6.28318530717958647692f));
  Value rr = b.create<math::SqrtOp>(
      loc, b.create<arith::MulFOp>(loc, negTwo,
                                   b.create<math::LogOp>(loc, u0)));
  Value z = b.create<arith::MulFOp>(
      loc, rr, b.create<math::CosOp>(
                   loc, b.create<arith::MulFOp>(loc, twoPi, u1)));

  // out[i] = y[i] - eta·grad[i] + noise_scale·z
  Value yv = b.create<memref::LoadOp>(loc, Y, ValueRange{gid});
  Value gv = b.create<memref::LoadOp>(loc, G, ValueRange{gid});
  Value step = b.create<arith::SubFOp>(
      loc, yv, b.create<arith::MulFOp>(loc, eta, gv));
  Value res = b.create<arith::AddFOp>(
      loc, step, b.create<arith::MulFOp>(loc, noiseScale, z));
  b.create<memref::StoreOp>(loc, res, OUT, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
  (void)idxTy;
}

struct GenerateROCMEbmLangevinKernelPass
    : PassWrapper<GenerateROCMEbmLangevinKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMEbmLangevinKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-ebm-langevin-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.ebm_langevin directive into a Langevin-step "
           "gpu kernel with on-device Philox-4x32-10 Box-Muller noise";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.ebm_langevin")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.ebm_langevin missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type i32 = b.getIntegerType(32);
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType(
          {memF32, memF32, idxTy, f32, f32, i32, i32, i32, i32, i32, i32,
           memF32},
          {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitLangevinBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMEbmLangevinKernelPass() {
  return std::make_unique<GenerateROCMEbmLangevinKernelPass>();
}
