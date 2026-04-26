
// EffectAnnotationPass.cpp
//
// Infers the side-effect class of each func.func in the module by walking its
// body ops, then attaches  tessera.effect = "pure"|"random"|"memory"|"io"  as
// a function attribute.
//
// Effect lattice (least → most permissive):
//   pure (0) < random (1) < memory (2) < io (3)
//
// Inference rules:
//   tessera.flash_attn  with dropout_p != 0.0   → random
//   tessera.copy                                 → memory
//   any arg  tessera.effect = "write"|"reduce_*" → memory
//   func.call to an external non-tessera func    → io
//   everything else                              → pure
//
// Validation:
//   If a func already carries  tessera.effect = "pure"  AND the body infers
//   a higher effect level, the pass emits an error and signals failure.
//   This enforces the  @jit(deterministic=True)  contract from Phase 1.

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

enum class EffectLevel : int { Pure = 0, Random = 1, Memory = 2, IO = 3 };

static EffectLevel maxEffect(EffectLevel a, EffectLevel b) {
  return (static_cast<int>(a) >= static_cast<int>(b)) ? a : b;
}

static StringRef effectStr(EffectLevel e) {
  switch (e) {
  case EffectLevel::Pure:   return "pure";
  case EffectLevel::Random: return "random";
  case EffectLevel::Memory: return "memory";
  case EffectLevel::IO:     return "io";
  }
  return "pure";
}

static EffectLevel parseEffectStr(StringRef s) {
  if (s == "random") return EffectLevel::Random;
  if (s == "memory") return EffectLevel::Memory;
  if (s == "io")     return EffectLevel::IO;
  return EffectLevel::Pure;
}

struct EffectAnnotation
    : public PassWrapper<EffectAnnotation, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EffectAnnotation)

  StringRef getArgument()    const override { return "tessera-effect-annotation"; }
  StringRef getDescription() const override {
    return "Infer and annotate tessera.effect on each func.func";
  }

  // Infer the effect of a single op (not recursive).
  static EffectLevel inferOpEffect(Operation *op) {
    StringRef name = op->getName().getStringRef();

    // flash_attn with non-zero dropout is non-deterministic.
    if (name == "tessera.flash_attn") {
      if (auto dp = op->getAttrOfType<FloatAttr>("dropout_p"))
        if (dp.getValueAsDouble() != 0.0)
          return EffectLevel::Random;
      return EffectLevel::Pure;
    }

    // Explicit copy/store has memory side-effects.
    if (name == "tessera.copy") return EffectLevel::Memory;

    // External function calls (not tessera.*) raise the level to IO.
    if (name == "func.call") {
      if (auto calleeAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee")) {
        StringRef callee = calleeAttr.getValue();
        if (!callee.starts_with("tessera"))
          return EffectLevel::IO;
      }
      return EffectLevel::Memory;
    }

    return EffectLevel::Pure;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod.walk([&](func::FuncOp func) {
      // Read the pre-existing annotation (set by @jit decorator).
      StringRef priorStr;
      if (auto ea = func.getAttrOfType<StringAttr>("tessera.effect"))
        priorStr = ea.getValue();

      // Infer from argument region annotations.
      EffectLevel inferred = EffectLevel::Pure;
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        auto ea = func.getArgAttrOfType<StringAttr>(i, "tessera.effect");
        if (!ea) continue;
        StringRef mode = ea.getValue();
        if (mode == "write" || mode.starts_with("reduce_"))
          inferred = maxEffect(inferred, EffectLevel::Memory);
      }

      // Walk every op in the function body.
      func.walk([&](Operation *op) {
        inferred = maxEffect(inferred, inferOpEffect(op));
      });

      // Validate against a "pure" declaration contract.
      if (priorStr == "pure" && inferred > EffectLevel::Pure) {
        func.emitError()
            << "function '" << func.getName()
            << "' is declared deterministic/pure but body contains "
            << effectStr(inferred) << " effects";
        signalPassFailure();
        return;
      }

      // If the function was already annotated with a higher effect level
      // (e.g. from a callee analysis earlier in the pipeline), keep it.
      EffectLevel prior =
          priorStr.empty() ? EffectLevel::Pure : parseEffectStr(priorStr);
      EffectLevel final_ = maxEffect(prior, inferred);

      func.setAttr("tessera.effect",
                   StringAttr::get(func.getContext(), effectStr(final_)));
    });
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createEffectAnnotationPass() {
  return std::make_unique<EffectAnnotation>();
}
} // namespace tessera
