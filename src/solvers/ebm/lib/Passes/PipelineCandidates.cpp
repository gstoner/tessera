//===- PipelineCandidates.cpp ---------------------------------*- C++ -*-===//
//
// EBMPipelineCandidatesPass: walks `tessera_ebm.self_verify` ops, finds
// the producing `tessera_ebm.decode_init` (which carries the K
// candidate count), and attaches pipeline-parallelism annotations that
// a backend codegen pass uses to map the K-candidate axis across
// streams / devices.
//
// The K axis is the natural parallelism for self_verify (each
// candidate runs an independent inner-loop chain, only joining at the
// final reduce).  This pass marks the chain so codegen can:
//   - place each k ∈ [0, K) on a different CUDA stream / MTLDevice queue;
//   - emit overlapping H2D + compute + D2H scheduling;
//   - allocate K independent state buffers up-front.
//
// Attributes attached:
//   tessera.ebm.pipeline_K          : i64  on self_verify and decode_init
//   tessera.ebm.pipeline_axis       : str  "k" (the axis to parallelize)
//   tessera.ebm.pipeline_target_ref : symbol-ref of the matching
//                                     decode_init (back-link for codegen)
//
// V1 supports the common case where a `decode_init` and `self_verify`
// share a control-flow path within the same function.  More general
// pairing (across functions / regions) is deferred.
//
//===----------------------------------------------------------------------===//

#include "tessera/EBM/EBMPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kDecodeInitOpName = "tessera_ebm.decode_init";
constexpr StringRef kSelfVerifyOpName = "tessera_ebm.self_verify";
constexpr StringRef kPipelineKAttr = "tessera.ebm.pipeline_K";
constexpr StringRef kPipelineAxisAttr = "tessera.ebm.pipeline_axis";
constexpr StringRef kPipelinedMarkerAttr = "tessera.ebm.pipelined";

struct EBMPipelineCandidatesPass
    : public PassWrapper<EBMPipelineCandidatesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EBMPipelineCandidatesPass)

  StringRef getArgument() const final {
    return "tessera-ebm-pipeline-candidates";
  }
  StringRef getDescription() const final {
    return "Map the K-candidate axis of ebm.decode_init + ebm.self_verify "
           "chains across streams / devices via pipeline annotations.";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    mod.walk([&](func::FuncOp fn) {
      // Collect decode_init ops in function order for back-linking.
      SmallVector<Operation *, 4> decodeInits;
      fn.walk([&](Operation *op) {
        if (op->getName().getStringRef() == kDecodeInitOpName) {
          decodeInits.push_back(op);
        }
      });
      if (decodeInits.empty()) return;

      fn.walk([&](Operation *op) {
        if (op->getName().getStringRef() != kSelfVerifyOpName) return;
        // Find the most-recent decode_init that precedes this
        // self_verify (v1: prefer the dominator-tree closest match;
        // use op order as a simple proxy).
        Operation *partner = nullptr;
        for (Operation *di : decodeInits) {
          if (di->isBeforeInBlock(op) ||
              di->getBlock() == op->getBlock() == false) {
            partner = di;
          }
        }
        if (!partner) {
          // No earlier decode_init found.  Skip — the self_verify
          // consumes externally-supplied candidates; codegen can still
          // pipeline if it wants but we don't have a K binding.
          return;
        }
        auto K = partner->getAttrOfType<IntegerAttr>("K");
        if (!K) return;

        op->setAttr(kPipelineKAttr, K);
        op->setAttr(kPipelineAxisAttr, StringAttr::get(ctx, "k"));
        op->setAttr(kPipelinedMarkerAttr, builder.getUnitAttr());

        partner->setAttr(kPipelineKAttr, K);
        partner->setAttr(kPipelineAxisAttr, StringAttr::get(ctx, "k"));
        partner->setAttr(kPipelinedMarkerAttr, builder.getUnitAttr());
      });
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEBMPipelineCandidatesPass() {
  return std::make_unique<EBMPipelineCandidatesPass>();
}

}  // namespace tessera
