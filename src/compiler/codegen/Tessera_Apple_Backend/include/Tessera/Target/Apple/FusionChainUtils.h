//===- FusionChainUtils.h - Shared fusion chain-walk helpers ---*- C++ -*-===//
//
// Workstream A2b (COMPILER_REFACTOR_PLAN §3, Workstream A). The matmul-chain
// fusion passes (matmul→softmax / →gelu / →rmsnorm, and matmul→softmax→matmul)
// each hand-rolled the same two mechanics: reading the compiler's fusion intent
// (Decision #19) and walking a chain consumer up to its producer while enforcing
// the single-use fusion-safety invariant and the Decision #21 descriptor/IR
// mismatch warning. Hoisted here as one definition.
//
// Composite pre-fused passes (SwiGLU / MLADecode / NSA) don't walk SSA chains,
// so they don't use these.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_FUSIONCHAINUTILS_H
#define TESSERA_TARGET_APPLE_FUSIONCHAINUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace tessera {
namespace apple {

/// Decision #19 — the compiler stamps `tessera.fusion.intent` on the chain-tail
/// op it intends to fuse.  Returns true when `op` carries that intent naming
/// `expectedKernel` (so the rewrite is descriptor-driven rather than
/// structurally re-discovered).
inline bool fusionDescriptorDriven(::mlir::Operation *op,
                                   ::mlir::StringRef expectedKernel) {
  if (auto a = op->getAttrOfType<::mlir::StringAttr>("tessera.fusion.intent"))
    return a.getValue() == expectedKernel;
  return false;
}

/// Walk from a chain consumer's `operand` up to the producer op that defines it,
/// enforcing that the producer is `expectedProducer` and that the intermediate
/// has exactly one use (fusion safety).  Returns the producer op, or a
/// match-failure if the operand has no defining op, the producer is the wrong
/// op, or the intermediate is used elsewhere.  On a descriptor/IR disagreement
/// (`descriptorDriven` but the producer is wrong), emits the Decision #21
/// warning before failing.
inline ::mlir::FailureOr<::mlir::Operation *> walkChainProducer(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *consumer,
    ::mlir::Value operand, ::mlir::StringRef expectedProducer,
    bool descriptorDriven) {
  ::mlir::Operation *def = operand.getDefiningOp();
  if (!def)
    return rewriter.notifyMatchFailure(
        consumer, "fusion: chain operand has no defining op");
  if (def->getName().getStringRef() != expectedProducer) {
    if (descriptorDriven)
      consumer->emitWarning(
          "tessera.fusion.intent set but the chain operand is not from '" +
          expectedProducer.str() +
          "' — descriptor/IR mismatch; falling back to unfused");
    return rewriter.notifyMatchFailure(
        consumer, "fusion: chain producer is not the expected op");
  }
  if (!operand.hasOneUse())
    return rewriter.notifyMatchFailure(
        consumer, "fusion: chain producer result has multiple uses");
  return def;
}

} // namespace apple
} // namespace tessera

#endif // TESSERA_TARGET_APPLE_FUSIONCHAINUTILS_H
