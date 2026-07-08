//===- GenerateROCMEbmAffineLangevinKernel.cpp - affine Langevin step ----===//
//
// Expands a `tessera_rocm.ebm_affine_langevin` directive into a Langevin-step
// gpu kernel that takes its Gaussian noise AS AN INPUT (host-drawn), rather than
// drawing its own Philox noise — the manifold half of the P7 EBM follow-up.
// `tessera.ebm.{bivector,sphere}_langevin_step` project the gradient and a
// host-drawn, grade-projected Gaussian onto a manifold subspace, then take the
// same affine combination as the Philox lane. One thread per element `i`:
//
//   out[i] = y[i] - eta·grad[i] + noise_scale·noise[i]
//
// Mirrors the Apple-GPU `_try_apple_gpu_langevin_step_f32` bridge (noise as an
// operand) so the manifold samplers match their numpy reference exactly (the
// host noise, not device Philox). CPU analog: tessera_x86_ebm_affine_langevin_f32.
// Args: (y : memref<?xf32>, grad : memref<?xf32>, noise : memref<?xf32>,
// n : index, eta f32, noise_scale f32, out : memref<?xf32>).
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitAffineLangevinBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type idxTy = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value Y = f.getArgument(0), G = f.getArgument(1), NOISE = f.getArgument(2);
  Value N = f.getArgument(3), eta = f.getArgument(4);
  Value noiseScale = f.getArgument(5), OUT = f.getArgument(6);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  // out[i] = y[i] - eta·grad[i] + noise_scale·noise[i]
  Value yv = b.create<memref::LoadOp>(loc, Y, ValueRange{gid});
  Value gv = b.create<memref::LoadOp>(loc, G, ValueRange{gid});
  Value zv = b.create<memref::LoadOp>(loc, NOISE, ValueRange{gid});
  Value step = b.create<arith::SubFOp>(
      loc, yv, b.create<arith::MulFOp>(loc, eta, gv));
  Value res = b.create<arith::AddFOp>(
      loc, step, b.create<arith::MulFOp>(loc, noiseScale, zv));
  b.create<memref::StoreOp>(loc, res, OUT, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
  (void)idxTy;
}

struct GenerateROCMEbmAffineLangevinKernelPass
    : PassWrapper<GenerateROCMEbmAffineLangevinKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMEbmAffineLangevinKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-ebm-affine-langevin-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.ebm_affine_langevin directive into a "
           "Langevin-step gpu kernel taking host-drawn noise as an input";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.ebm_affine_langevin")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.ebm_affine_langevin missing name");
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
          {memF32, memF32, memF32, idxTy, f32, f32, memF32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitAffineLangevinBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMEbmAffineLangevinKernelPass() {
  return std::make_unique<GenerateROCMEbmAffineLangevinKernelPass>();
}
