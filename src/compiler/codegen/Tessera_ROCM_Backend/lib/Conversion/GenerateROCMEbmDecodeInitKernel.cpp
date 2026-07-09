//===- GenerateROCMEbmDecodeInitKernel.cpp - decode-init noise-apply -----===//
//
// Expands a `tessera_rocm.ebm_decode_init` directive into an elementwise
// noise-apply gpu kernel — the initialization half of the DFlash / EBM
// speculative-decode lane. When `tessera.ebm.decode_init(init_strategy="noise",
// mean=…)` seeds K candidate trajectories, it combines a host-drawn
// unit-variance Gaussian with a per-element base (mean) offset. One thread per
// element `i`:
//
//   out[i] = base[i] + std·noise[i]
//
// `base` and `noise` are already broadcast to the full trajectory shape by the
// caller, so this is a pure elementwise affine combine — no reduction, no RNG
// (the host draws the noise so the fast path and the numpy reference share
// identical Gaussian samples). Mirrors the Apple ebm_decode_init bridge
// (_try_apple_gpu_decode_init_noise_apply_f32). CPU analog:
// tessera_x86_ebm_decode_init_noise_apply_f32. Args:
// (base : memref<?xf32>, noise : memref<?xf32>, n : index, std f32,
// out : memref<?xf32>).
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

void emitDecodeInitBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value BASE = f.getArgument(0), NOISE = f.getArgument(1);
  Value N = f.getArgument(2), std = f.getArgument(3), OUT = f.getArgument(4);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  // out[i] = base[i] + std·noise[i]
  Value bv = b.create<memref::LoadOp>(loc, BASE, ValueRange{gid});
  Value zv = b.create<memref::LoadOp>(loc, NOISE, ValueRange{gid});
  Value res = b.create<arith::AddFOp>(
      loc, bv, b.create<arith::MulFOp>(loc, std, zv));
  b.create<memref::StoreOp>(loc, res, OUT, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMEbmDecodeInitKernelPass
    : PassWrapper<GenerateROCMEbmDecodeInitKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMEbmDecodeInitKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-ebm-decode-init-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.ebm_decode_init directive into an elementwise "
           "noise-apply (base + std·noise) gpu kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.ebm_decode_init")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.ebm_decode_init missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      // (base : memref<?xf32>, noise : memref<?xf32>, n : index, std f32,
      //  out : memref<?xf32>)
      auto fnTy = b.getFunctionType({memF32, memF32, idxTy, f32, memF32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitDecodeInitBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMEbmDecodeInitKernelPass() {
  return std::make_unique<GenerateROCMEbmDecodeInitKernelPass>();
}
