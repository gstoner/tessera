#include "Tessera/Transforms/SemanticEffects.h"

#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;

namespace tessera {

SemanticEffectLevel joinSemanticEffects(SemanticEffectLevel lhs,
                                        SemanticEffectLevel rhs) {
  return static_cast<int>(lhs) >= static_cast<int>(rhs) ? lhs : rhs;
}

llvm::StringRef stringifySemanticEffect(SemanticEffectLevel effect) {
  switch (effect) {
  case SemanticEffectLevel::Pure: return "pure";
  case SemanticEffectLevel::Random: return "random";
  case SemanticEffectLevel::Movement: return "movement";
  case SemanticEffectLevel::State: return "state";
  case SemanticEffectLevel::Collective: return "collective";
  case SemanticEffectLevel::Memory: return "memory";
  case SemanticEffectLevel::IO: return "io";
  case SemanticEffectLevel::Top: return "top";
  }
  return "top";
}

bool parseSemanticEffect(llvm::StringRef value,
                         SemanticEffectLevel &effect) {
  if (value == "pure") effect = SemanticEffectLevel::Pure;
  else if (value == "random") effect = SemanticEffectLevel::Random;
  else if (value == "movement") effect = SemanticEffectLevel::Movement;
  else if (value == "state") effect = SemanticEffectLevel::State;
  else if (value == "collective") effect = SemanticEffectLevel::Collective;
  else if (value == "memory") effect = SemanticEffectLevel::Memory;
  else if (value == "io") effect = SemanticEffectLevel::IO;
  else if (value == "top") effect = SemanticEffectLevel::Top;
  else return false;
  return true;
}

SemanticEffectLevel getRegisteredSemanticEffect(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("tessera.effect_kind")) {
    SemanticEffectLevel effect;
    if (!parseSemanticEffect(attr.getValue(), effect))
      return SemanticEffectLevel::Top;
    // A textual attribute may refine a registered side effect (memory write
    // into state/random/collective), but it cannot erase one by claiming pure.
    if (effect == SemanticEffectLevel::Pure && !isMemoryEffectFree(op))
      return SemanticEffectLevel::Top;
    return effect;
  }

  // MemoryEffectOpInterface is the registered fallback for non-Graph
  // dialects. `isMemoryEffectFree` also recognizes the Pure trait.
  if (isMemoryEffectFree(op))
    return SemanticEffectLevel::Pure;
  if (isa<MemoryEffectOpInterface>(op))
    return SemanticEffectLevel::Memory;
  return SemanticEffectLevel::Top;
}

bool isSemanticSchedulingBarrier(Operation *op) {
  if (op->getNumRegions() != 0)
    return true;

  if (auto alias = op->getAttrOfType<StringAttr>("tessera.aliasing"))
    if (alias.getValue() != "none")
      return true;
  if (isa<CastOpInterface, ViewLikeOpInterface>(op))
    return true;

  // Even explicitly keyed RNG retains a sample identity. Moving waits across
  // it is deferred until a consumer has a proof that the key/counter lineage
  // makes the transformation legal.
  if (auto stochastic =
          op->getAttrOfType<StringAttr>("tessera.stochastic_identity"))
    if (stochastic.getValue() != "none")
      return true;

  return getRegisteredSemanticEffect(op) != SemanticEffectLevel::Pure;
}

} // namespace tessera
