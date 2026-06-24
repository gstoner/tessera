//===- LowerROCMAsyncCopyToLoop.cpp - runnable async_copy lowering --------===//
//
// Lowers the ROCm staging seam to REAL, executable code instead of the
// artifact-only `*.contract` markers that TesseraTargetToROCDL emits:
//
//   tessera_rocm.async_copy(%dst, %src, %count)  ->  a cooperative copy loop
//       for i = tid; i < count; i += blockDim.x:  %dst[i] = %src[i]
//   tessera_rocm.wait(%tok)                       ->  gpu.barrier
//
// On RDNA there is no hardware global→LDS DMA (no GLOBAL_LOAD_LDS — confirmed
// from the RDNA3.5 ISA archive), so a "copy" is a cooperative load-from-global /
// store-to-LDS loop; the standard gpu.module → ROCDL lowering turns the
// memref.load/store into real `global_load` / `ds_store`. This is the runnable
// half of the Fork-A pipeline (the markers stay for the IR-contract path when
// this pass is not run). `%dst`/`%src` are 1-D memrefs of the same element type;
// `%count` is the element count. The token SSA edge is dropped here (the
// lowered `gpu.barrier` provides the ordering the token modeled).
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerROCMAsyncCopyToLoopPass
    : PassWrapper<LowerROCMAsyncCopyToLoopPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerROCMAsyncCopyToLoopPass)

  StringRef getArgument() const final { return "lower-rocm-async-copy"; }
  StringRef getDescription() const final {
    return "Lower tessera_rocm.async_copy to a runnable cooperative global→LDS "
           "copy loop (and tessera_rocm.wait to gpu.barrier) — the executable "
           "alternative to the artifact-only contract markers.";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> copies, waits;
    module.walk([&](Operation *op) {
      StringRef n = op->getName().getStringRef();
      if (n == "tessera_rocm.async_copy")
        copies.push_back(op);
      else if (n == "tessera_rocm.wait")
        waits.push_back(op);
    });

    // Waits first: each becomes a gpu.barrier and is erased — which releases the
    // token uses, so the async_copy results below are use-free and erasable.
    for (Operation *op : waits) {
      OpBuilder b(op);
      b.create<gpu::BarrierOp>(op->getLoc());
      op->erase();
    }

    for (Operation *op : copies) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      if (op->getNumOperands() < 3) {
        op->emitError("lower-rocm-async-copy: async_copy needs (dst, src, count)");
        return signalPassFailure();
      }
      Value dst = op->getOperand(0), src = op->getOperand(1);
      Value count = op->getOperand(2);
      auto dstTy = dyn_cast<MemRefType>(dst.getType());
      auto srcTy = dyn_cast<MemRefType>(src.getType());
      if (!dstTy || !srcTy ||
          dstTy.getElementType() != srcTy.getElementType()) {
        op->emitError("lower-rocm-async-copy: dst/src must be memrefs of the "
                      "same element type for the runnable lowering");
        return signalPassFailure();
      }
      if (!count.getType().isIndex())
        count = b.create<arith::IndexCastOp>(loc, b.getIndexType(), count);

      // Cooperative copy: for i = tid; i < count; i += blockDim.x.
      Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      Value bdim = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
      auto loop = b.create<scf::ForOp>(loc, tid, count, bdim);
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(loop.getBody());
        Value i = loop.getInductionVar();
        Value v = b.create<memref::LoadOp>(loc, src, ValueRange{i});
        b.create<memref::StoreOp>(loc, v, dst, ValueRange{i});
      }
      // The token result modeled completion ordering; the gpu.barrier (from the
      // consuming wait, lowered above) provides it. Any remaining token use
      // (e.g. an mma in a fused kernel — the Fork-A T3c case) is dropped so the
      // op erases cleanly; the barrier still serializes the copy before its use.
      for (Value r : op->getResults())
        r.dropAllUses();
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createLowerROCMAsyncCopyToLoopPass() {
  return std::make_unique<LowerROCMAsyncCopyToLoopPass>();
}
