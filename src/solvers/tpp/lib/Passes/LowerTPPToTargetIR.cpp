
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace {
struct LowerTPPToTargetIR : public PassWrapper<LowerTPPToTargetIR, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-tpp-to-target-ir"; }
  StringRef getDescription() const final { return "Lower TPP to Target-IR (sketch)"; }
  void runOnOperation() final {
    getOperation()->walk([&](Operation *op){
      if (op->getName().getStringRef().endswith("tpp.bc.enforce")) {
        // Attach a synthetic attribute marking masked store lowering occurred.
        op->setAttr("lowered.bc.masked", UnitAttr::get(op->getContext()));
      }
    });
  }
};
}
std::unique_ptr<Pass> createLowerTPPToTargetIRPass(){ return std::make_unique<LowerTPPToTargetIR>(); }
