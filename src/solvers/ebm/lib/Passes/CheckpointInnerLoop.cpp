//===- CheckpointInnerLoop.cpp ------------------------------------*- C++ -*-===//
//
// EBMCheckpointInnerLoopPass: walks every `scf.for` whose body contains
// `tessera_ebm.langevin_step` or `tessera_ebm.inner_step` ops, and
// attaches checkpointing annotations so a backend codegen pass can
// rematerialize the inner-loop trajectory instead of keeping every
// intermediate state live.
//
// Attributes attached:
//   tessera.ebm.checkpoint_budget : i64 — max number of live states
//                                    the codegen pass should keep
//                                    (default: 4, override via the
//                                    `--checkpoint-budget` pass option).
//   tessera.ebm.checkpoint_loop   : unit — present on the scf.for op.
//   tessera.ebm.recompute_step    : unit — present on each contained
//                                    langevin_step / inner_step op,
//                                    signaling "rematerializable".
//
// Note: this pass is **annotation-only**.  The rematerialization itself
// is done by Phase F2 rematerialize logic at the backend boundary.
//
//===----------------------------------------------------------------------===//

#include "tessera/EBM/EBMPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kLangevinStepOpName = "tessera_ebm.langevin_step";
constexpr StringRef kInnerStepOpName = "tessera_ebm.inner_step";
constexpr StringRef kCheckpointBudgetAttr = "tessera.ebm.checkpoint_budget";
constexpr StringRef kCheckpointLoopAttr = "tessera.ebm.checkpoint_loop";
constexpr StringRef kRecomputeStepAttr = "tessera.ebm.recompute_step";

static bool isInnerLoopStep(StringRef name) {
  return name == kLangevinStepOpName || name == kInnerStepOpName;
}

struct EBMCheckpointInnerLoopPass
    : public PassWrapper<EBMCheckpointInnerLoopPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EBMCheckpointInnerLoopPass)

  // Pass option: maximum number of live states to keep (rest are
  // rematerialized).  Default 4 — enough to fit a typical T=16 chain
  // with one checkpoint every 4 steps.
  Option<int64_t> checkpointBudget{
      *this, "budget",
      llvm::cl::desc("Maximum number of live inner-loop states"),
      llvm::cl::init(4)};

  StringRef getArgument() const final {
    return "tessera-ebm-checkpoint-inner-loop";
  }
  StringRef getDescription() const final {
    return "Annotate scf.for loops containing ebm.langevin_step / "
           "inner_step with checkpoint budgets for backend rematerialization.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    mod.walk([&](scf::ForOp loop) {
      // Walk the loop body for any inner-loop step ops.
      bool sawInnerStep = false;
      loop.getBody()->walk([&](Operation *op) {
        if (isInnerLoopStep(op->getName().getStringRef())) {
          sawInnerStep = true;
          op->setAttr(kRecomputeStepAttr, builder.getUnitAttr());
        }
      });
      if (sawInnerStep) {
        loop->setAttr(kCheckpointLoopAttr, builder.getUnitAttr());
        loop->setAttr(kCheckpointBudgetAttr,
                      builder.getI64IntegerAttr(checkpointBudget.getValue()));
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEBMCheckpointInnerLoopPass() {
  return std::make_unique<EBMCheckpointInnerLoopPass>();
}

}  // namespace tessera
