//===- DynamicTopologyPass.cpp — Dynamic topology management (Phase 7) -----===//
//
// Manages topology mutations that may occur at runtime (e.g., fault-tolerant
// reshaping, adaptive mesh refinement, node failure/recovery).
//
// Pass actions
// ------------
//
// 1. SCAN for topology.create ops whose `kind` attribute is NOT a static
//    compile-time constant (contains "dynamic", "adaptive", or "fault").
//    These are "mutable topologies".
//
// 2. For each mutable topology:
//    a. Set "topology.dynamic" = true on the topology.create op.
//    b. Insert a "fence" annotation ("topology.fence" = true) on every op
//       in the same block that reads the topology handle — this tells the
//       scheduler that a barrier is required before each consumer.
//    c. Insert a "replan" hook annotation ("topology.replan" = true) on the
//       topology.create op so a runtime callback can be wired in.
//
// 3. SCAN for halo.exchange ops that carry `policy = "adaptive"`.
//    These may remap neighbor ranks at runtime; annotate with:
//      "topology.adaptive_halo" = true
//      "topology.replan_hook"   = "adaptive_halo_replan"
//
// The pass does NOT emit new ops — topology management is architecture-
// specific.  All information is recorded as attributes for a backend pass.
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

/// Return true when a topology kind string signals dynamic / adaptive topology.
static bool isMutableKind(StringRef kind) {
  return kind.contains("dynamic") || kind.contains("adaptive") ||
         kind.contains("fault")   || kind.contains("custom_graph");
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct DynamicTopologyPass
    : public PassWrapper<DynamicTopologyPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DynamicTopologyPass)

  StringRef getArgument() const final { return "tessera-topology-dynamic"; }
  StringRef getDescription() const final {
    return "Insert fence tokens and replan hooks for dynamic topology changes";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder(ctx);

    // ----------------------------------------------------------------
    // Step 1: find all mutable topology.create ops.
    // ----------------------------------------------------------------
    llvm::SmallVector<Operation *> mutableTopos;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() !=
          "tessera.neighbors.topology.create")
        return WalkResult::advance();

      StringRef kind;
      if (auto ka = op->getAttrOfType<StringAttr>("kind"))
        kind = ka.getValue();
      else if (auto da = op->getAttr("kind"))
        kind = "unknown";

      if (isMutableKind(kind))
        mutableTopos.push_back(op);
      return WalkResult::advance();
    });

    // ----------------------------------------------------------------
    // Step 2: annotate mutable topology ops and their consumers.
    // ----------------------------------------------------------------
    for (Operation *topoOp : mutableTopos) {
      topoOp->setAttr("topology.dynamic",
                      builder.getBoolAttr(true));
      topoOp->setAttr("topology.replan",
                      builder.getBoolAttr(true));
      topoOp->setAttr("topology.replan_hook",
                      builder.getStringAttr("tessera_topology_replan"));

      // Annotate every user of the topology value.
      if (topoOp->getNumResults() > 0) {
        for (Operation *user : topoOp->getResult(0).getUsers()) {
          if (!user->hasAttr("topology.fence"))
            user->setAttr("topology.fence", builder.getBoolAttr(true));
          // Tag neighbour.read ops with a runtime delta-check requirement.
          if (user->getName().getStringRef() ==
              "tessera.neighbors.neighbor.read") {
            user->setAttr("topology.runtime_delta_check",
                          builder.getBoolAttr(true));
          }
        }
      }
    }

    // ----------------------------------------------------------------
    // Step 3: handle adaptive halo.exchange ops.
    // ----------------------------------------------------------------
    mod.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef() !=
          "tessera.neighbors.halo.exchange")
        return WalkResult::advance();

      auto policy = op->getAttrOfType<StringAttr>("policy");
      if (!policy || policy.getValue() != "adaptive")
        return WalkResult::advance();

      op->setAttr("topology.adaptive_halo",
                  builder.getBoolAttr(true));
      op->setAttr("topology.replan_hook",
                  builder.getStringAttr("adaptive_halo_replan"));
      return WalkResult::advance();
    });

    // ----------------------------------------------------------------
    // Step 4: mark static topologies as verified-static for downstream.
    // ----------------------------------------------------------------
    mod.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef() !=
          "tessera.neighbors.topology.create")
        return WalkResult::advance();
      if (!op->hasAttr("topology.dynamic"))
        op->setAttr("topology.static", builder.getBoolAttr(true));
      return WalkResult::advance();
    });
  }
};

void registerDynamicTopologyPass() {
  PassRegistration<DynamicTopologyPass>();
}

} // namespace neighbors
} // namespace tessera
