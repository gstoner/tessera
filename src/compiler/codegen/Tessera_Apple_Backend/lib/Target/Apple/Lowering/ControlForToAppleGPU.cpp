//===- ControlForToAppleGPU.cpp - Lower tessera.control_for -------------===//
//
// Phase-G G-B — lower the Graph-IR bounded loop `tessera.control_for` to the
// Apple Target-IR op `tessera_apple.gpu.control_loop` (value-preserving: the
// iter-args feed in, the final carry feeds out, so the rewrite is a plain
// replaceOp). IR-only — the Decision #19 hardware-free Target IR layer. The
// recorded `symbol` (`tessera_apple_gpu_run_graph_loop_f32`) is the runtime
// capability that actually executes the loop (exercised today via the GraphFn
// lane; an MLIR-driven executor off this op is a follow-on).
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace ::mlir;

namespace tessera {
namespace apple {
namespace {

constexpr llvm::StringLiteral kControlLoopOp = "tessera_apple.gpu.control_loop";
constexpr llvm::StringLiteral kRunGraphLoopSymbol =
    "tessera_apple_gpu_run_graph_loop_f32";

struct LowerControlForToAppleGPUPass
    : public PassWrapper<LowerControlForToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerControlForToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-control-for-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.control_for (bounded loop) to the Apple GPU "
           "tessera_apple.gpu.control_loop Target-IR op (Phase-G G-B)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  // A manual module walk (not the greedy driver) — the greedy driver's region
  // simplification would DCE unrelated result-less artifact ops (e.g. an
  // unlowered `tile.cholesky`) before sibling passes lower them. This pass only
  // touches `tessera.control_for` ops; everything else is left untouched.
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::SmallVector<Operation *> targets;
    int64_t ordinal = 0;
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.control_for")
        targets.push_back(op);
      else if (name == kControlLoopOp)
        ++ordinal;  // continue numbering after any pre-existing control_loops
    });
    OpBuilder builder(&getContext());
    for (Operation *op : targets) {
      auto body = op->getAttrOfType<FlatSymbolRefAttr>("body");
      auto start = op->getAttrOfType<IntegerAttr>("start");
      auto stop = op->getAttrOfType<IntegerAttr>("stop");
      auto step = op->getAttrOfType<IntegerAttr>("step");
      if (!body || !start || !stop || !step) {
        op->emitError("tessera.control_for needs body/start/stop/step attributes");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(op);
      OperationState st(op->getLoc(), kControlLoopOp);
      st.addOperands(op->getOperands());
      st.addTypes(op->getResultTypes());
      st.addAttribute("body", body);
      st.addAttribute("start", start);
      st.addAttribute("stop", stop);
      st.addAttribute("step", step);
      st.addAttribute("ordinal", builder.getI64IntegerAttr(ordinal++));
      st.addAttribute("status", builder.getStringAttr("artifact"));
      st.addAttribute("symbol", builder.getStringAttr(kRunGraphLoopSymbol));
      st.addAttribute("framework", builder.getStringAttr("MPSGraph"));
      Operation *loop = builder.create(st);
      op->replaceAllUsesWith(loop);
      op->erase();
    }
  }
};

// Standalone registration so the pass is invokable as
// `--tessera-control-for-to-apple_gpu` (the lit fixture runs it in isolation).
static PassRegistration<LowerControlForToAppleGPUPass> gReg;

} // namespace

std::unique_ptr<Pass> createLowerControlForToAppleGPUPass() {
  return std::make_unique<LowerControlForToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
