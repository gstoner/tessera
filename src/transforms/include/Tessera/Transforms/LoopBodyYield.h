#pragma once

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"

namespace tessera {

// Close a loop/branch body block that the caller just created.
//
// `scf::ForOp::create` installs a default `scf.yield` only when the loop
// carries no iteration arguments; with `initArgs` it deliberately leaves the
// body block empty so the caller supplies the yielded values.  Probing that
// state with `Block::getTerminator()` is not a legal test: it asserts on
// `mightHaveTerminator()` under an assertions-enabled MLIR, and dereferences an
// empty operation list under NDEBUG, where the same call is silently undefined
// behavior rather than an abort -- so a release CI build cannot observe the
// defect at all.  `Block::mightHaveTerminator()` is total over both shapes of
// block and is the only safe way to ask.
inline void closeBodyWithYield(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Block *body, mlir::ValueRange results) {
  if (body->mightHaveTerminator()) {
    if (auto existing =
            mlir::dyn_cast<mlir::scf::YieldOp>(body->getTerminator())) {
      existing.getResultsMutable().assign(results);
      return;
    }
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(body);
  mlir::scf::YieldOp::create(builder, loc, results);
}

} // namespace tessera
