#include "Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace {
struct LowerPowerToTile : public PassWrapper<LowerPowerToTile, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
};
struct LowerPowerToTarget : public PassWrapper<LowerPowerToTarget, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
};
}
namespace tessera { namespace power {
std::unique_ptr<mlir::Pass> createLowerPowerToTilePass() { return std::make_unique<LowerPowerToTile>(); }
std::unique_ptr<mlir::Pass> createLowerPowerToTargetPass() { return std::make_unique<LowerPowerToTarget>(); }
void registerPowerPasses() {
  PassRegistration<LowerPowerToTile>("tessera-lower-power-to-tile", "Lower power.attn to Tile IR");
  PassRegistration<LowerPowerToTarget>("tessera-lower-power-to-target", "Lower power.attn to Target IR");
}}}
