#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera { namespace neighbors {

struct HaloInferPass : public PassWrapper<HaloInferPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaloInferPass)
  StringRef getArgument() const final { return "tessera-halo-infer"; }
  StringRef getDescription() const final { return "Infer halo widths from Δ-uses"; }
  void runOnOperation() override {
    // TODO: scan neighbor.read deltas and stencil taps, compute per-axis max |Δ|,
    // attach as attributes on halo.region or stencil.apply users.
  }
};

void registerHaloInferPass() { PassRegistration<HaloInferPass>(); }

}} // namespace
