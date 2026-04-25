
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
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
      if (name.endswith("tpp.grad")) {
        if (!op->hasAttr(HaloAttrName))
          op->setAttr(HaloAttrName, StringAttr::get(op->getContext(), "1,1,0"));
      } else if (name.endswith("tpp.stencil.apply")) {
        if (Attribute r = op->getAttr("radius")) {
          if (!op->hasAttr(HaloAttrName))
            op->setAttr(HaloAttrName, cast<IntegerAttr>(r).getValue().toString(10) + StringRef(",") + "?,?");
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
