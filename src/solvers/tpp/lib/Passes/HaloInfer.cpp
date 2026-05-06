
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
using namespace mlir;

static StringRef HaloAttrName = "tpp.halo";

namespace {
struct HaloInfer : public PassWrapper<HaloInfer, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-halo-infer"; }
  StringRef getDescription() const final { return "Infer halos from tpp.grad/tpp.stencil.apply usage"; }
  void runOnOperation() final {
    ModuleOp m = getOperation();
    m.walk([&](Operation *op){
      StringRef name = op->getName().getStringRef();
      // Very simple heuristic: if grad => halo (1,1,0); if stencil.apply => read 'radius' attr if present.
      if (name.ends_with("tpp.grad")) {
        if (!op->hasAttr(HaloAttrName))
          op->setAttr(HaloAttrName, StringAttr::get(op->getContext(), "1,1,0"));
      } else if (name.ends_with("tpp.stencil.apply")) {
        if (Attribute r = op->getAttr("radius")) {
          if (!op->hasAttr(HaloAttrName))
            op->setAttr(HaloAttrName, StringAttr::get(
                op->getContext(),
                (Twine(cast<IntegerAttr>(r).getInt()) + ",?,?").str()));
        } else {
          if (!op->hasAttr(HaloAttrName))
            op->setAttr(HaloAttrName, StringAttr::get(op->getContext(), "1,1,1"));
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createHaloInferPass(){ return std::make_unique<HaloInfer>(); }
