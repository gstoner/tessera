//===- EffectAnnotationPass.cpp - registered Graph effect analysis --------===//
//
// W2.2 derives function effects from canonical operation contracts and MLIR
// interfaces. It never guesses from operation-name substrings. Internal call
// summaries reach a fixed point; external calls and unknown operations fail
// closed. The resulting function attribute is consumed by differentiation,
// collective insertion, and scheduling legality.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "Tessera/Transforms/SemanticEffects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

using tessera::SemanticEffectLevel;

struct FunctionFacts {
  SemanticEffectLevel local = SemanticEffectLevel::Pure;
  SmallVector<func::FuncOp> callees;
};

struct EffectAnnotation
    : public PassWrapper<EffectAnnotation, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EffectAnnotation)

  StringRef getArgument() const override { return "tessera-effect-annotation"; }
  StringRef getDescription() const override {
    return "Derive function effects from registered Graph semantics and MLIR "
           "operation interfaces";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    DenseMap<Operation *, FunctionFacts> facts;
    DenseMap<Operation *, SemanticEffectLevel> summaries;

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      FunctionFacts functionFacts;
      for (unsigned index = 0; index < function.getNumArguments(); ++index) {
        auto attr = function.getArgAttrOfType<StringAttr>(index, "tessera.effect");
        if (!attr)
          continue;
        StringRef mode = attr.getValue();
        if (mode == "write" || mode.starts_with("reduce_"))
          functionFacts.local = tessera::joinSemanticEffects(
              functionFacts.local, SemanticEffectLevel::Memory);
      }

      function.walk([&](Operation *op) {
        if (op == function.getOperation())
          return;
        if (auto call = dyn_cast<func::CallOp>(op)) {
          if (auto callee = symbols.lookup<func::FuncOp>(call.getCallee())) {
            if (callee.isDeclaration())
              functionFacts.local = tessera::joinSemanticEffects(
                  functionFacts.local, SemanticEffectLevel::IO);
            else
              functionFacts.callees.push_back(callee);
          } else
            functionFacts.local = tessera::joinSemanticEffects(
                functionFacts.local, SemanticEffectLevel::IO);
          return;
        }
        functionFacts.local = tessera::joinSemanticEffects(
            functionFacts.local, tessera::getRegisteredSemanticEffect(op));
      });
      summaries[function.getOperation()] = functionFacts.local;
      facts[function.getOperation()] = std::move(functionFacts);
    }

    // Monotone finite lattice: at most seven promotions per function. This
    // handles recursion and call cycles without relying on source order.
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &entry : facts) {
        SemanticEffectLevel next = entry.second.local;
        for (func::FuncOp callee : entry.second.callees)
          next = tessera::joinSemanticEffects(
              next, summaries.lookup(callee.getOperation()));
        if (next != summaries.lookup(entry.first)) {
          summaries[entry.first] = next;
          changed = true;
        }
      }
    }

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      SemanticEffectLevel inferred = summaries.lookup(function.getOperation());
      StringRef prior;
      if (auto attr = function->getAttrOfType<StringAttr>("tessera.effect"))
        prior = attr.getValue();

      if (prior == "pure" && inferred != SemanticEffectLevel::Pure) {
        function.emitError()
            << "function '" << function.getName()
            << "' is declared deterministic/pure but registered Graph IR "
               "contains "
            << tessera::stringifySemanticEffect(inferred) << " effects";
        signalPassFailure();
        continue;
      }

      SemanticEffectLevel declared = SemanticEffectLevel::Pure;
      if (!prior.empty() && !tessera::parseSemanticEffect(prior, declared))
        declared = SemanticEffectLevel::Top;
      SemanticEffectLevel finalEffect =
          tessera::joinSemanticEffects(declared, inferred);
      function->setAttr(
          "tessera.effect",
          StringAttr::get(function.getContext(),
                          tessera::stringifySemanticEffect(finalEffect)));
    }
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createEffectAnnotationPass() {
  return std::make_unique<EffectAnnotation>();
}
} // namespace tessera
