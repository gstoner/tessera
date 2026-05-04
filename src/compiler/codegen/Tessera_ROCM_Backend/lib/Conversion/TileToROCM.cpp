#include "TesseraROCM/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerTileToROCMPass
    : PassWrapper<LowerTileToROCMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToROCMPass)

  StringRef getArgument() const final { return "lower-tile-to-rocm"; }
  StringRef getDescription() const final {
    return "Lower Tessera Tile IR matmul movement contracts to ROCm Target IR";
  }

  void runOnOperation() override {
    SmallVector<Operation *> worklist;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.mma" || name == "tile.async_copy" ||
          name == "tile.wait_async" || name == "tile.kv_cache" ||
          name.starts_with("tile.tmem."))
        worklist.push_back(op);
    });

    Value lastAsyncToken;
    for (Operation *op : worklist) {
      OpBuilder builder(op);
      StringRef name = op->getName().getStringRef();

      if (name == "tile.mma") {
        if (op->getNumOperands() < 2 || op->getNumResults() != 1) {
          op->emitError("ROCm lowering requires tile.mma(lhs, rhs) -> result");
          signalPassFailure();
          return;
        }

        OperationState state(op->getLoc(), "tessera_rocm.mfma");
        // The v1 contract carries a scalar accumulator operand. Until Tile IR
        // models explicit accumulator SSA, use lhs as the artifact accumulator.
        state.addOperands({op->getOperand(0), op->getOperand(1),
                           op->getOperand(0)});
        state.addTypes(op->getResultTypes());
        state.addAttribute("arch", builder.getStringAttr("gfx90a"));
        state.addAttribute("shape", builder.getStringAttr("m16n16k16"));
        state.addAttribute("accum", builder.getStringAttr("f32"));
        state.addAttribute("source", builder.getStringAttr("tessera.matmul"));
        state.addAttribute("ordinal", builder.getI64IntegerAttr(0));
        Operation *rocmOp = builder.create(state);
        op->replaceAllUsesWith(rocmOp->getResults());
        op->erase();
        continue;
      }

      if (name == "tile.async_copy") {
        if (op->getNumOperands() < 3 || op->getNumResults() != 1) {
          op->emitError("ROCm lowering requires tile.async_copy(dst, src, bytes) -> token");
          signalPassFailure();
          return;
        }

        OperationState state(op->getLoc(), "tessera_rocm.async_copy");
        state.addOperands({op->getOperand(0), op->getOperand(1),
                           op->getOperand(2)});
        state.addTypes(op->getResultTypes());
        state.addAttribute("src_space", builder.getStringAttr("global"));
        state.addAttribute("dst_space", builder.getStringAttr("lds"));
        state.addAttribute("arch", builder.getStringAttr("gfx90a"));
        Operation *rocmOp = builder.create(state);
        lastAsyncToken = rocmOp->getResult(0);
        op->replaceAllUsesWith(rocmOp->getResults());
        op->erase();
        continue;
      }

      if (name == "tile.wait_async") {
        if (!lastAsyncToken) {
          op->emitError("ROCm lowering requires tile.wait_async after tile.async_copy");
          signalPassFailure();
          return;
        }

        OperationState state(op->getLoc(), "tessera_rocm.wait");
        state.addOperands(lastAsyncToken);
        builder.create(state);
        op->erase();
        continue;
      }

      if (name == "tile.kv_cache") {
        op->emitError("ROCm lowering does not implement KV-cache artifacts in this phase");
        signalPassFailure();
        return;
      }

      if (name.starts_with("tile.tmem.")) {
        op->emitError("ROCm lowering does not support TMEM operations");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerTileToROCMImpl() {
  return std::make_unique<LowerTileToROCMPass>();
}
