//===- DecomposeCompositeOpsPass.cpp - target_verify / ntk_rope rewrite ---===//
//
// Factory for the standalone `tessera-decompose-composite-ops` pass. The
// patterns and the pass body live in CompositeDecomposition.h so the Apple
// backend library -- which links neither the Tessera dialect nor this library
// -- runs the same source (ODS triage WIRE slice 1, Decision #31).
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/CompositeDecomposition.h"
#include "Tessera/Transforms/Passes.h"

namespace tessera {
std::unique_ptr<mlir::Pass> createDecomposeCompositeOpsPass() {
  return std::make_unique<composite::DecomposeCompositeOpsPass>();
}
} // namespace tessera
