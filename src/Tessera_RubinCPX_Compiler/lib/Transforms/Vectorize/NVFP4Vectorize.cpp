
#include "tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
namespace {
struct NVFP4VectorizePass
    : public PassWrapper<NVFP4VectorizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NVFP4VectorizePass)
  StringRef getArgument() const override { return "tessera-vectorize-nvfp4"; }
  StringRef getDescription() const override {
    return "Legalize matmul/attention tiles to NVFP4 MMA forms with FP16/FP32 accumulators";
  }
  void runOnOperation() override {
    // TODO: Populate patterns against matmul/attn ops; map to packed NVFP4 types.
  }
};
}

namespace tessera {
std::unique_ptr<mlir::Pass> createNVFP4VectorizePass() {
  return std::make_unique<NVFP4VectorizePass>();
}
} // namespace tessera
