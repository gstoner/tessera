// Materialize Graph layout casts into NVIDIA operand-binding contracts.

#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace tessera {
namespace {

struct NVIDIAGraphLayoutMaterializationPass
    : public PassWrapper<NVIDIAGraphLayoutMaterializationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NVIDIAGraphLayoutMaterializationPass)

  StringRef getArgument() const override {
    return "tessera-nvidia-materialize-layout-casts";
  }
  StringRef getDescription() const override {
    return "Consume Graph layout casts as NVIDIA staging-layout contracts";
  }

  void runOnOperation() override {
    SmallVector<Operation *> casts;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.cast" &&
          op->hasAttr("tessera.layout"))
        casts.push_back(op);
    });

    static const llvm::StringSet<> supported = {
        "row_major", "col_major", "bhsd", "nhwc"};
    for (Operation *cast : casts) {
      auto layout = cast->getAttrOfType<StringAttr>("tessera.layout");
      if (!layout || !supported.contains(layout.getValue())) {
        cast->emitError("NVIDIA Graph layout materializer does not support '")
            << (layout ? layout.getValue() : StringRef("<missing>")) << "'";
        signalPassFailure();
        return;
      }
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast->emitError("NVIDIA Graph layout materializer requires unary cast");
        signalPassFailure();
        return;
      }
      auto source = cast->getAttrOfType<StringAttr>("tessera.source_layout");
      for (OpOperand &use : llvm::make_early_inc_range(
               cast->getResult(0).getUses())) {
        Operation *consumer = use.getOwner();
        std::string suffix = llvm::Twine(use.getOperandNumber()).str();
        consumer->setAttr("tessera.nvidia.operand_layout_" + suffix, layout);
        if (source)
          consumer->setAttr("tessera.nvidia.source_layout_" + suffix, source);
      }
      cast->getResult(0).replaceAllUsesWith(cast->getOperand(0));
      cast->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createNVIDIAGraphLayoutMaterializationPass() {
  return std::make_unique<NVIDIAGraphLayoutMaterializationPass>();
}

} // namespace tessera
