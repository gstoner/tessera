//===- LowerScheduleToTarget.cpp (v1.4) ------------------------------------===//
//
// HONEST SCAFFOLD — NOT a working lowering.
//
// The real Schedule -> Tile -> Target lowering in Tessera is performed by the
// **Python compiler spine** (`tessera.compiler.schedule_ir`,
// `tessera.compiler.tile_ir`, `tessera.compiler.target_ir`), which is what
// `@tessera.jit` uses and what the unit + lit suites exercise. That path already
// emits the queue / async-copy / barrier / launch-metadata ops this file's old
// pseudocode merely described in comments.
//
// This C++ pass was never registered in `tessera-opt` or any named pipeline, and
// its body was an empty no-op that silently *succeeded* — i.e. it could be wired
// into a pipeline and appear to "lower scheduling" while doing nothing. That is
// exactly the overstated-claim gap the compiler-layer remediation plan targets
// (docs/audit/compiler/COMPILER_AUDIT.md, item G2). Rather than ship a
// redundant, untested, unregistered C++ reimplementation of the Python spine,
// the pass now FAILS LOUDLY if it is ever invoked, so it can never masquerade as
// a successful lowering.
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace tessera { namespace schedule {

struct LowerScheduleToTargetPass
    : public PassWrapper<LowerScheduleToTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerScheduleToTargetPass)

  StringRef getArgument() const final { return "tessera-lower-schedule-to-target"; }
  StringRef getDescription() const final {
    return "SCAFFOLD ONLY — not implemented in C++; the working Schedule->Target "
           "lowering is the Python compiler spine. Fails if invoked.";
  }

  void runOnOperation() override {
    // No silent no-op: an unimplemented lowering must never report success.
    getOperation().emitError(
        "tessera-lower-schedule-to-target is a scaffold and is not implemented "
        "in C++. The Schedule->Tile->Target lowering is performed by the Python "
        "compiler spine (tessera.compiler.{schedule_ir,tile_ir,target_ir}); do "
        "not wire this pass into a pipeline.");
    signalPassFailure();
  }
};

std::unique_ptr<Pass> createLowerScheduleToTargetPass() {
  return std::make_unique<LowerScheduleToTargetPass>();
}

}} // namespace tessera::schedule
