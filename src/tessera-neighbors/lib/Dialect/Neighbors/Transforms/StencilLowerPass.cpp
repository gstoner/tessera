#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera { namespace neighbors {

struct StencilLowerPass : public PassWrapper<StencilLowerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StencilLowerPass)
  StringRef getArgument() const final { return "tessera-stencil-lower"; }
  StringRef getDescription() const final { return "Lower stencil.apply to loops/kernels + exchanges"; }
  void runOnOperation() override {
    // TODO: expand stencil taps into compute, insert halo.exchange, pack/unpack skeletons.
  }
};

void registerStencilLowerPass() { PassRegistration<StencilLowerPass>(); }

}} // namespace
