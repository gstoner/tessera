// Materialize Graph layout casts into Apple runtime operand contracts.

#include "Tessera/Target/Apple/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace tessera::apple {
namespace {

struct MaterializeGraphLayoutToApplePass
    : public PassWrapper<MaterializeGraphLayoutToApplePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeGraphLayoutToApplePass)

  StringRef getArgument() const override {
    return "tessera-apple-materialize-layout-casts";
  }
  StringRef getDescription() const override {
    return "Consume Graph layout casts as Apple runtime binding contracts";
  }

  void runOnOperation() override {
    SmallVector<Operation *> casts;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.cast" &&
          op->hasAttr("tessera.layout"))
        casts.push_back(op);
    });

    static const llvm::StringSet<> supported = {
        "row_major", "bhsd", "nhwc"};
    for (Operation *cast : casts) {
      auto layout = cast->getAttrOfType<StringAttr>("tessera.layout");
      if (!layout || !supported.contains(layout.getValue())) {
        cast->emitError("Apple Graph layout materializer does not support '")
            << (layout ? layout.getValue() : StringRef("<missing>")) << "'";
        signalPassFailure();
        return;
      }
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast->emitError("Apple Graph layout materializer requires unary cast");
        signalPassFailure();
        return;
      }
      auto source = cast->getAttrOfType<StringAttr>("tessera.source_layout");
      for (OpOperand &use : llvm::make_early_inc_range(
               cast->getResult(0).getUses())) {
        Operation *consumer = use.getOwner();
        std::string suffix = llvm::Twine(use.getOperandNumber()).str();
        consumer->setAttr("tessera.apple.operand_layout_" + suffix, layout);
        if (source)
          consumer->setAttr("tessera.apple.source_layout_" + suffix, source);
      }
      cast->getResult(0).replaceAllUsesWith(cast->getOperand(0));
      cast->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createMaterializeGraphLayoutToApplePass() {
  return std::make_unique<MaterializeGraphLayoutToApplePass>();
}

} // namespace tessera::apple
