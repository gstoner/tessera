#include "Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace {
// LLVM 23's `PassRegistration<T>()` takes its argument and description from
// the pass itself; the two-argument constructor this file used is gone, and the
// type ids are required for a PassWrapper. Both passes are still scaffolds
// (empty bodies), as the surface manifest records.
struct LowerPowerToTile : public PassWrapper<LowerPowerToTile, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPowerToTile)
  StringRef getArgument() const final { return "tessera-lower-power-to-tile"; }
  StringRef getDescription() const final { return "Lower power.attn to Tile IR"; }
  void runOnOperation() override {}
};
struct LowerPowerToTarget : public PassWrapper<LowerPowerToTarget, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPowerToTarget)
  StringRef getArgument() const final { return "tessera-lower-power-to-target"; }
  StringRef getDescription() const final { return "Lower power.attn to Target IR"; }
  void runOnOperation() override {}
};
}
namespace tessera { namespace power {
std::unique_ptr<mlir::Pass> createLowerPowerToTilePass() { return std::make_unique<LowerPowerToTile>(); }
std::unique_ptr<mlir::Pass> createLowerPowerToTargetPass() { return std::make_unique<LowerPowerToTarget>(); }
void registerPowerPasses() {
  PassRegistration<LowerPowerToTile>();
  PassRegistration<LowerPowerToTarget>();
}}}
