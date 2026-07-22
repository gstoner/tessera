//===- FusionPattern.h - Declarative Apple fusion registry -----*- C++ -*-===//

#ifndef TESSERA_TARGET_APPLE_FUSIONPATTERN_H
#define TESSERA_TARGET_APPLE_FUSIONPATTERN_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace tessera::apple {

using FusionRewriteFn = ::mlir::LogicalResult (*)(
    ::mlir::Operation *, ::mlir::PatternRewriter &);

/// One declarative row in the Apple runtime-fusion registry.  Matching order is
/// represented by `benefit`; the generic rewrite owns the MLIR mechanics while
/// the handler owns only the family-specific ABI/shape materialization.
struct FusionPattern {
  ::llvm::StringRef family;
  ::llvm::StringRef rootOp;
  unsigned benefit;
  FusionRewriteFn rewrite;
};

::mlir::LogicalResult rewriteMatmulSoftmaxFusion(
    ::mlir::Operation *, ::mlir::PatternRewriter &);
::mlir::LogicalResult rewriteMatmulSoftmaxMatmulFusion(
    ::mlir::Operation *, ::mlir::PatternRewriter &);
::mlir::LogicalResult rewriteMatmulGeluFusion(
    ::mlir::Operation *, ::mlir::PatternRewriter &);
::mlir::LogicalResult rewriteMatmulRMSNormFusion(
    ::mlir::Operation *, ::mlir::PatternRewriter &);
::mlir::LogicalResult rewriteSwigluFusion(
    ::mlir::Operation *, ::mlir::PatternRewriter &);
::mlir::LogicalResult rewriteMLADecodeFusion(
    ::mlir::Operation *, ::mlir::PatternRewriter &);
::mlir::LogicalResult rewriteNativeSparseAttnFusion(
    ::mlir::Operation *, ::mlir::PatternRewriter &);

::llvm::ArrayRef<FusionPattern> appleFusionPatterns();
void populateDeclarativeAppleFusionPatterns(::mlir::RewritePatternSet &patterns);

} // namespace tessera::apple

#endif // TESSERA_TARGET_APPLE_FUSIONPATTERN_H
