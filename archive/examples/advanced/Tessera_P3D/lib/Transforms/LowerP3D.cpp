#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

using namespace mlir;

namespace {
struct LowerP3DPass : public PassWrapper<LowerP3DPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerP3DPass)
  StringRef getArgument() const final { return "tessera-lower-p3d"; }
  StringRef getDescription() const final { return "Lower P3D ops to Tessera Tile/Target IR."; }
  void runOnOperation() override {
    // TODO: Implement legalization to Tile IR conv/attention and then to Target IR.
  }
};
} // namespace

std::unique_ptr<Pass> createLowerP3DPass() { return std::make_unique<LowerP3DPass>(); }
