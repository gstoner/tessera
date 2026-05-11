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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class LightningAttnFusionPass
    : public PassWrapper<LightningAttnFusionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LightningAttnFusionPass)

  StringRef getArgument() const override { return "tessera-lightning-attn-fusion"; }
  StringRef getDescription() const override {
    return "Prepare/fuse tessera.lightning_attention for tiled backend lowering";
  }
  void runOnOperation() override {}
};

class DeltaAttnChunkingPass
    : public PassWrapper<DeltaAttnChunkingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeltaAttnChunkingPass)

  StringRef getArgument() const override { return "tessera-delta-attn-chunking"; }
  StringRef getDescription() const override {
    return "Lower Gated DeltaNet/Kimi Delta attention into chunked scan form";
  }
  void runOnOperation() override {}
};

class HybridAttnExpandPass
    : public PassWrapper<HybridAttnExpandPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HybridAttnExpandPass)

  StringRef getArgument() const override { return "tessera-hybrid-attn-expand"; }
  StringRef getDescription() const override {
    return "Expand named Ling/Kimi hybrid attention policies into primitive attention ops";
  }
  void runOnOperation() override {}
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

} // namespace tessera
