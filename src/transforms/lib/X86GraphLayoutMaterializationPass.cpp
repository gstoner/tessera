// Materialize Graph layout casts into x86 operand-binding contracts.
//
// The generic x86 emitter owns an executable row-major binding ABI through
// ExecutableLayout/materialize_layouts. This pass is the Graph-side bridge:
// consume a requested row/column-major cast, preserve source-layout provenance on the
// exact consumer operand, and remove the same-type marker before tiling.

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

struct X86GraphLayoutMaterializationPass
    : public PassWrapper<X86GraphLayoutMaterializationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      X86GraphLayoutMaterializationPass)

  StringRef getArgument() const override {
    return "tessera-x86-materialize-layout-casts";
  }
  StringRef getDescription() const override {
    return "Consume Graph layout casts as executable x86 binding "
           "contracts";
  }

  void runOnOperation() override {
    SmallVector<Operation *> casts;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.cast" &&
          op->hasAttr("tessera.layout"))
        casts.push_back(op);
    });

    // BHSD/NHWC are C-contiguous physical bindings in the generic x86 emitter;
    // rank-2 column-major is a distinct Fortran-order pointer-stride ABI.
    static const llvm::StringSet<> supported = {
        "row_major", "col_major", "bhsd", "nhwc"};
    for (Operation *cast : casts) {
      auto layout = cast->getAttrOfType<StringAttr>("tessera.layout");
      if (!layout || !supported.contains(layout.getValue())) {
        cast->emitError("x86 Graph layout materializer does not support '")
            << (layout ? layout.getValue() : StringRef("<missing>")) << "'";
        signalPassFailure();
        return;
      }
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast->emitError("x86 Graph layout materializer requires unary cast");
        signalPassFailure();
        return;
      }
      auto source = cast->getAttrOfType<StringAttr>("tessera.source_layout");
      for (OpOperand &use :
           llvm::make_early_inc_range(cast->getResult(0).getUses())) {
        Operation *consumer = use.getOwner();
        std::string suffix = llvm::Twine(use.getOperandNumber()).str();
        consumer->setAttr("tessera.x86.operand_layout_" + suffix, layout);
        if (source)
          consumer->setAttr("tessera.x86.source_layout_" + suffix, source);
      }
      cast->getResult(0).replaceAllUsesWith(cast->getOperand(0));
      cast->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createX86GraphLayoutMaterializationPass() {
  return std::make_unique<X86GraphLayoutMaterializationPass>();
}

} // namespace tessera
