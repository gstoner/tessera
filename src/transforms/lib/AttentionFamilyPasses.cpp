//===- AttentionFamilyPasses.cpp - Modern attention pass slots -*- C++ -*-===//
//
// Reference/compiler-visibility pass slots for Lightning Attention, DeltaNet /
// Kimi Delta chunking, and named hybrid attention policy expansion. The first
// implementation is intentionally conservative: expose stable pass names and
// pipeline positions while preserving IR. Backend-specific rewrites can be
// layered onto these pass entrypoints without changing user-facing pipelines.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

void markReasoningVisible(Operation *op, OpBuilder &builder, StringRef family,
                          StringRef variant) {
  op->setAttr("tessera.reasoning.compiler_visible",
              builder.getBoolAttr(true));
  op->setAttr("tessera.reasoning.family", builder.getStringAttr(family));
  op->setAttr("tessera.reasoning.variant", builder.getStringAttr(variant));
}

StringRef hybridVariant(Operation *op) {
  if (auto pattern = op->getAttrOfType<StringAttr>("pattern"))
    return pattern.getValue();
  return "hybrid_attention";
}

class LightningAttnFusionPass
    : public PassWrapper<LightningAttnFusionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LightningAttnFusionPass)

  StringRef getArgument() const override { return "tessera-lightning-attn-fusion"; }
  StringRef getDescription() const override {
    return "Prepare/fuse tessera.lightning_attention for tiled backend lowering";
  }
  void runOnOperation() override {
    OpBuilder builder(getOperation().getContext());
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.lightning_attention")
        markReasoningVisible(op, builder, "lightning", "lightning_attention");
    });
  }
};

class DeltaAttnChunkingPass
    : public PassWrapper<DeltaAttnChunkingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeltaAttnChunkingPass)

  StringRef getArgument() const override { return "tessera-delta-attn-chunking"; }
  StringRef getDescription() const override {
    return "Lower Gated DeltaNet/Kimi Delta attention into chunked scan form";
  }
  void runOnOperation() override {
    OpBuilder builder(getOperation().getContext());
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.gated_deltanet")
        markReasoningVisible(op, builder, "delta", "gated_deltanet");
      else if (name == "tessera.kimi_delta_attention")
        markReasoningVisible(op, builder, "delta", "kimi_delta_attention");
      else if (name == "tessera.modified_delta_attention")
        markReasoningVisible(op, builder, "delta", "modified_delta_attention");
    });
  }
};

class HybridAttnExpandPass
    : public PassWrapper<HybridAttnExpandPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HybridAttnExpandPass)

  StringRef getArgument() const override { return "tessera-hybrid-attn-expand"; }
  StringRef getDescription() const override {
    return "Expand named Ling/Kimi hybrid attention policies into primitive attention ops";
  }
  void runOnOperation() override {
    OpBuilder builder(getOperation().getContext());
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tessera.hybrid_attention")
        return;
      markReasoningVisible(op, builder, "hybrid", hybridVariant(op));
    });
  }
};

// Lookahead Sparse Attention (LSA) — experimental, inference-only composite
// attention policy.  Like HybridAttnExpandPass this is a compiler-visibility
// slot: it preserves the semantic op while marking it for the sparse-attention
// backend lane (host-mediated block selection + GPU dense attention).  See
// docs/audit/domain/archive/lsa_scope.md (D1-D5).
class LookaheadSparseAttnExpandPass
    : public PassWrapper<LookaheadSparseAttnExpandPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LookaheadSparseAttnExpandPass)

  StringRef getArgument() const override {
    return "tessera-lookahead-sparse-attn-expand";
  }
  StringRef getDescription() const override {
    return "Mark tessera.lookahead_sparse_attention for the sparse-attention "
           "backend lane (local window ∪ selected historical blocks)";
  }
  void runOnOperation() override {
    OpBuilder builder(getOperation().getContext());
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tessera.lookahead_sparse_attention")
        return;
      markReasoningVisible(op, builder, "lookahead_sparse",
                           "lookahead_sparse_attention");
    });
  }
};

// Gap-2 follow-on — LSA Graph→Schedule lowering. For each
// tessera.lookahead_sparse_attention op, emit a schedule.prefetch op that makes
// the cold-pool KV staging a first-class IR value: the prefetch consumes the
// LSA op's K operand and the LSA op is rewired to consume the prefetch result,
// so the dataflow reads "stage K (from the host cold pool) → attend over it".
//
// The prefetch is emitted with into="host" + overlap="none": it *records* the
// staging tier without claiming overlap, matching the synchronous Gap-1 runtime
// staging. A backend that can asynchronously stage would set overlap="compute",
// at which point the (already real) tpp-async-prefetch pass software-pipelines
// it. Running -tessera-lookahead-sparse-prefetch then -tpp-async-prefetch is the
// end-to-end Graph→Schedule→overlap flow.
class LookaheadSparsePrefetchPass
    : public PassWrapper<LookaheadSparsePrefetchPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LookaheadSparsePrefetchPass)

  StringRef getArgument() const override {
    return "tessera-lookahead-sparse-prefetch";
  }
  StringRef getDescription() const override {
    return "Emit schedule.prefetch{into=host} recording LSA cold-pool KV "
           "staging (Graph->Schedule lowering)";
  }
  void runOnOperation() override {
    OpBuilder builder(getOperation().getContext());
    llvm::SmallVector<Operation *> lsaOps;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.lookahead_sparse_attention")
        lsaOps.push_back(op);
    });
    for (Operation *op : lsaOps) {
      if (op->getNumOperands() < 2)
        continue; // need at least (Q, K)
      if (op->hasAttr("tessera.lsa.prefetch_emitted"))
        continue; // idempotent
      Value kv = op->getOperand(1); // K — the tensor staged from the cold pool
      builder.setInsertionPoint(op);
      OperationState state(op->getLoc(), "schedule.prefetch");
      state.addOperands(kv);
      state.addTypes(kv.getType());
      state.addAttribute("into", builder.getStringAttr("host"));
      state.addAttribute("overlap", builder.getStringAttr("none"));
      state.addAttribute("tessera.lsa.staging",
                         builder.getStringAttr("host_cold_pool"));
      Operation *prefetch = builder.create(state);
      // Rewire the LSA op to consume the staged value — a true dataflow edge
      // (prefetch must precede its consumer; the async-prefetch dependency
      // guard then keeps the prefetch ahead of the attention).
      op->setOperand(1, prefetch->getResult(0));
      op->setAttr("tessera.lsa.prefetch_emitted", builder.getBoolAttr(true));
    }
  }
};

} // namespace

namespace tessera {

std::unique_ptr<Pass> createLightningAttnFusionPass() {
  return std::make_unique<LightningAttnFusionPass>();
}

std::unique_ptr<Pass> createDeltaAttnChunkingPass() {
  return std::make_unique<DeltaAttnChunkingPass>();
}

std::unique_ptr<Pass> createHybridAttnExpandPass() {
  return std::make_unique<HybridAttnExpandPass>();
}

std::unique_ptr<Pass> createLookaheadSparseAttnExpandPass() {
  return std::make_unique<LookaheadSparseAttnExpandPass>();
}

std::unique_ptr<Pass> createLookaheadSparsePrefetchPass() {
  return std::make_unique<LookaheadSparsePrefetchPass>();
}

} // namespace tessera
