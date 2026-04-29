//===- PipelineOverlapPass.cpp — Halo-compute overlap (Phase 7) ------------===//
//
// Inserts async overlap of halo exchange and stencil compute.
//
// Strategy
// --------
// For each tessera.neighbors.stencil.apply (or halo.exchange) op in the IR,
// the pass assigns distinct "stream" tokens so that a backend can schedule
// communication on a dedicated stream while compute runs on another.
//
// Concretely the pass attaches these attributes:
//
//   On halo.exchange ops:
//     "comm.stream_id"  : I64Attr  — communication stream slot (0-based)
//     "comm.priority"   : I64Attr  — scheduling priority (higher = earlier)
//     "comm.async"      : BoolAttr true
//
//   On stencil.apply ops:
//     "compute.stream_id" : I64Attr — compute stream slot
//     "compute.depends_comm" : BoolAttr — must sync comm before compute
//
//   On pipeline.config ops:
//     "pipeline.resolved" : BoolAttr true (sentinel)
//     "pipeline.comm_stream" : I64Attr — the assigned comm stream id
//     "pipeline.compute_stream" : I64Attr — the assigned compute stream id
//
// Double-buffering
// ----------------
// When pipeline.config carries `double_buffer = true` the pass assigns
// alternating buffer indices (0/1) to successive stencil.apply ops within
// the same function and records them on each op as "pipeline.buffer_idx".
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {
namespace neighbors {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct PipelineConfig {
  int64_t  stages       = 1;
  bool     doubleBuffer = false;
  StringRef order       = "forward";
  StringRef overlap     = "lazy";
  StringRef reuse       = "none";
};

static PipelineConfig readConfig(Operation *configOp) {
  PipelineConfig cfg;
  if (!configOp) return cfg;
  if (auto a = configOp->getAttrOfType<IntegerAttr>("stages"))
    cfg.stages = a.getInt();
  if (auto a = configOp->getAttrOfType<BoolAttr>("double_buffer"))
    cfg.doubleBuffer = a.getValue();
  if (auto a = configOp->getAttrOfType<StringAttr>("order"))
    cfg.order = a.getValue();
  if (auto a = configOp->getAttrOfType<StringAttr>("overlap"))
    cfg.overlap = a.getValue();
  if (auto a = configOp->getAttrOfType<StringAttr>("reuse"))
    cfg.reuse = a.getValue();
  return cfg;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct PipelineOverlapPass
    : public PassWrapper<PipelineOverlapPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineOverlapPass)

  StringRef getArgument() const final { return "tessera-pipeline-overlap"; }
  StringRef getDescription() const final {
    return "Insert async halo-exchange / compute overlap via stream assignments";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder(ctx);

    // Process per-function so stream IDs reset across function boundaries.
    mod.walk([&](FuncOp func) {
      // Find the pipeline.config for this function (if any).
      Operation *configOp = nullptr;
      func.walk([&](Operation *op) {
        if (op->getName().getStringRef() ==
            "tessera.neighbors.pipeline.config") {
          configOp = op;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      PipelineConfig cfg = readConfig(configOp);
      bool doOverlap = (cfg.overlap != "none");

      // Comm stream = 0, compute stream = 1 (simple two-stream model).
      const int64_t commStream    = 0;
      const int64_t computeStream = 1;

      // Resolve pipeline.config itself.
      if (configOp && !configOp->hasAttr("pipeline.resolved")) {
        configOp->setAttr("pipeline.resolved",   builder.getBoolAttr(true));
        configOp->setAttr("pipeline.comm_stream",
                          builder.getI64IntegerAttr(commStream));
        configOp->setAttr("pipeline.compute_stream",
                          builder.getI64IntegerAttr(computeStream));
      }

      // Annotate halo.exchange ops.
      int64_t commPriority = 0;
      func.walk([&](Operation *op) -> WalkResult {
        if (op->getName().getStringRef() !=
            "tessera.neighbors.halo.exchange")
          return WalkResult::advance();
        if (op->hasAttr("comm.stream_id")) return WalkResult::advance();

        op->setAttr("comm.stream_id",
                    builder.getI64IntegerAttr(commStream));
        op->setAttr("comm.priority",
                    builder.getI64IntegerAttr(commPriority++));
        op->setAttr("comm.async", builder.getBoolAttr(doOverlap));
        return WalkResult::advance();
      });

      // Annotate stencil.apply ops (and assign double-buffer indices).
      int64_t applyIdx = 0;
      func.walk([&](Operation *op) -> WalkResult {
        if (op->getName().getStringRef() !=
            "tessera.neighbors.stencil.apply")
          return WalkResult::advance();

        op->setAttr("compute.stream_id",
                    builder.getI64IntegerAttr(computeStream));
        op->setAttr("compute.depends_comm",
                    builder.getBoolAttr(doOverlap));

        if (cfg.doubleBuffer) {
          op->setAttr("pipeline.buffer_idx",
                      builder.getI64IntegerAttr(applyIdx % 2));
        }
        ++applyIdx;
        return WalkResult::advance();
      });

      return WalkResult::advance();
    });
  }
};

void registerPipelineOverlapPass() {
  PassRegistration<PipelineOverlapPass>();
}

} // namespace neighbors
} // namespace tessera
