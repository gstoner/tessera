//===- TileValueProvenance.h — loop-carry-crossing value resolution -------===//
//
// P1a / CAKE Phase 1 §5.3 (2026-08-15): the shared resolver that lets Tile
// legality checks derive facts across `scf.for` block-argument edges instead
// of stopping at `getDefiningOp()` — the measured fail-open
// (compiler_enhancement.md §5.1.1: the canonical pipelined kernel shape was
// invisible to every legality pass because a block argument has no defining
// op).
//
// Semantics: a value is resolved to the set of non-block-argument values that
// can reach it. For an `scf.for` iter_arg BOTH sources are followed — the init
// operand (first iteration) and the yield operand (back edge) — so a
// loop-carried barrier resolves to its `tile.mbarrier.init` and a loop-carried
// token to its producing arrive/copy. Resolution is fail-closed: it reports
// incomplete when any path escapes derivation (a function argument, an
// unsupported region op), and callers must treat incomplete resolution as
// unproven, not as permission (Decision #30).
//
// One implementation per boundary (Decision #31): both WarpSpecLegalityPass
// and TileDataflowLegalityPass include this header; do not fork the walk.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TRANSFORMS_TILEVALUEPROVENANCE_H
#define TESSERA_TRANSFORMS_TILEVALUEPROVENANCE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace tessera {
namespace provenance {

struct Resolution {
  llvm::SmallPtrSet<mlir::Value, 4> roots; // non-block-arg values reached
  bool complete = true; // false => some path escaped derivation (fail closed)
};

inline void resolveImpl(mlir::Value v, Resolution &out,
                        llvm::SmallPtrSetImpl<mlir::Value> &visited) {
  if (!visited.insert(v).second)
    return; // back-edge fixed point
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    auto forOp =
        mlir::dyn_cast_or_null<mlir::scf::ForOp>(arg.getOwner()->getParentOp());
    if (!forOp || arg.getArgNumber() == 0) {
      // Function argument, non-scf.for region, or the induction variable:
      // no data provenance derivable here.
      out.complete = false;
      return;
    }
    unsigned idx = arg.getArgNumber() - 1; // skip induction variable
    resolveImpl(forOp.getInitArgs()[idx], out, visited);
    resolveImpl(forOp.getBody()->getTerminator()->getOperand(idx), out,
                visited);
    return;
  }
  out.roots.insert(v);
}

// Resolve `v` across loop-carry edges. `roots` holds every non-block-argument
// value that can flow into `v`; `complete` is false when any path could not be
// derived.
inline Resolution resolveThroughLoopCarry(mlir::Value v) {
  Resolution out;
  llvm::SmallPtrSet<mlir::Value, 8> visited;
  resolveImpl(v, out, visited);
  return out;
}

} // namespace provenance
} // namespace tessera

#endif // TESSERA_TRANSFORMS_TILEVALUEPROVENANCE_H
