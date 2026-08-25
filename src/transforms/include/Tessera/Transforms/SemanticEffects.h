#pragma once

#include "mlir/IR/Operation.h"
#include <string>
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

// W4-EFFECTS-1. Why an operation's recorded product does not verify, or the
// empty string when it does. The carrier is checked as a CHAIN — a supported
// schema, a lowercase 64-hex digest, a payload whose sha256 IS that digest,
// and a payload naming this operation and this effect class — so a
// fabricated digest, a missing payload, or a product copied from another
// operation are all refused. ONE implementation, shared by the paired pass
// and the region replayability walk, so admission cannot diverge between a
// top-level op and the same op nested in supported control flow (#31).
std::string recordedProductFailure(mlir::Operation *op,
                                   llvm::StringRef requiredClass);

// True when `op` carries a verified recorded product of `requiredClass`.
bool carriesVerifiedRecordedProduct(mlir::Operation *op,
                                    llvm::StringRef requiredClass);

// True when moving asynchronous synchronization across `op` would require an
// alias, mutation, stochastic, region, or ordering proof not present in IR.
bool isSemanticSchedulingBarrier(mlir::Operation *op);

} // namespace tessera
