#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

using namespace mlir;

namespace {
struct AutotuneP3DPass : public PassWrapper<AutotuneP3DPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutotuneP3DPass)
  StringRef getArgument() const final { return "tessera-autotune-p3d"; }
  StringRef getDescription() const final { return "Attach autotuning spaces for P3D pipelines."; }
  void runOnOperation() override {
    // TODO: Attach attributes for tile sizes, thread mappings, pipeline stages.
  }
};
}

std::unique_ptr<Pass> createAutotuneP3DPass() { return std::make_unique<AutotuneP3DPass>(); }
