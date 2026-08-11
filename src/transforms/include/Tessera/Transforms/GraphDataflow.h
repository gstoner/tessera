#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace tessera {

/// A fail-closed product fact attached to one Graph-IR SSA value.
///
/// `shapeTop` and `aliasTop` are independent: a dynamically-shaped value may
/// still have precise alias provenance, while a statically-shaped value
/// produced by an unregistered operation has unknown alias provenance.
struct GraphValueFact {
  bool initialized = false;
  bool shapeTop = true;
  llvm::SmallVector<int64_t, 4> shape;
  bool aliasTop = true;
  llvm::SmallVector<mlir::Value, 2> aliasRoots;

  static GraphValueFact top();
  static GraphValueFact join(const GraphValueFact &lhs,
                             const GraphValueFact &rhs);
  bool operator==(const GraphValueFact &rhs) const;
  void print(llvm::raw_ostream &os) const;
};

/// W2.1's shared Graph-IR analysis boundary.
///
/// The sparse product facts and whole-function liveness run on MLIR's
/// DataFlowSolver. Memory-dependence and activity queries consume those facts
/// plus registered operation effects. Unknown operations, regions, effects,
/// aliases, or stale state always produce a conservative answer.
class GraphDataflowAnalysis {
public:
  using ActiveOpSet = llvm::SmallPtrSet<mlir::Operation *, 32>;

  explicit GraphDataflowAnalysis(mlir::func::FuncOp function);
  ~GraphDataflowAnalysis();
  GraphDataflowAnalysis(GraphDataflowAnalysis &&) noexcept;
  GraphDataflowAnalysis &operator=(GraphDataflowAnalysis &&) noexcept;

  mlir::LogicalResult run();
  mlir::LogicalResult recompute();
  void invalidate();
  bool isValid() const;

  GraphValueFact getValueFact(mlir::Value value) const;
  bool isLive(mlir::Value value) const;
  bool mayAlias(mlir::Value lhs, mlir::Value rhs) const;
  bool hasMemoryDependence(mlir::Operation *lhs,
                           mlir::Operation *rhs) const;

  /// Whether `moving` may cross `across`. This is intentionally stronger than
  /// memory independence: stochastic identity, ordered collectives, alias
  /// construction, nested regions, and direct SSA dependence are barriers.
  bool canMoveAcross(mlir::Operation *moving,
                     mlir::Operation *across) const;

  /// Reverse activity from explicit result roots. `tessera.stop_gradient`
  /// remains active itself but blocks propagation to its operand.
  ActiveOpSet computeActivity(mlir::ValueRange roots) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace tessera
