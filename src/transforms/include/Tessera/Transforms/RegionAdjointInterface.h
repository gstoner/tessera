#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace tessera {

/// A cotangent for a value captured implicitly by a structured region.
///
/// RegionBranchOpInterface exposes control-flow successors, but SCF captures
/// are not operation operands. Returning positional operand cotangents would
/// therefore silently lose gradients for values closed over by an scf.if or
/// loop body. This pair is the canonical W4 boundary.
struct RegionCotangent {
  mlir::Value primal;
  mlir::Value cotangent;
};

/// Callback used by a control-flow model to transpose one selected region.
/// `captures` is deterministic and contains only differentiable values.
using RegionPullbackBuilder = llvm::function_ref<mlir::LogicalResult(
    mlir::Region &region, mlir::ValueRange outputCotangents,
    mlir::ValueRange blockArgumentValues,
    llvm::ArrayRef<mlir::Value> captures,
    const llvm::DenseMap<mlir::Value, mlir::Value> &savedValues,
    mlir::OpBuilder &builder,
    llvm::SmallVectorImpl<mlir::Value> &blockArgumentCotangents,
    llvm::SmallVectorImpl<mlir::Value> &captureCotangents)>;

/// Compiler-owned reverse-mode interface for structured control flow.
///
/// This deliberately models external SCF operations instead of adding Tessera
/// traits to another dialect. Unsupported region shapes fail closed. Models
/// may replay primal work, but must preserve the executed control path and
/// return cotangents for every differentiable implicit capture.
class RegionAdjointInterface {
public:
  static bool supports(mlir::Operation *op);

  static mlir::LogicalResult buildAdjoint(
      mlir::Operation *op, mlir::OpBuilder &builder,
      mlir::ValueRange outputCotangents,
      mlir::ValueRange explicitResiduals,
      RegionPullbackBuilder buildRegionPullback,
      llvm::SmallVectorImpl<RegionCotangent> &captureCotangents);
};

} // namespace tessera
