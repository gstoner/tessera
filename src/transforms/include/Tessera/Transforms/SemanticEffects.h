#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace tessera {

// W2.2 canonical semantic-effect lattice.  Operation producers may attach a
// registered `tessera.effect_kind`; otherwise the analysis consults MLIR's
// operation interfaces and fails closed for unregistered behavior.
enum class SemanticEffectLevel : int {
  Pure = 0,
  Random = 1,
  Movement = 2,
  State = 3,
  Collective = 4,
  Memory = 5,
  IO = 6,
  Top = 7,
};

SemanticEffectLevel joinSemanticEffects(SemanticEffectLevel lhs,
                                        SemanticEffectLevel rhs);
llvm::StringRef stringifySemanticEffect(SemanticEffectLevel effect);
bool parseSemanticEffect(llvm::StringRef value,
                         SemanticEffectLevel &effect);
SemanticEffectLevel getRegisteredSemanticEffect(mlir::Operation *op);

// True when moving asynchronous synchronization across `op` would require an
// alias, mutation, stochastic, region, or ordering proof not present in IR.
bool isSemanticSchedulingBarrier(mlir::Operation *op);

} // namespace tessera
