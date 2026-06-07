//===- ControlWhileToAppleGPU.cpp - Lower tessera.control_while ---------===//
//
// Phase-G close-out D — lower the Graph-IR bounded while `tessera.control_while`
// to the Apple Target-IR op `tessera_apple.gpu.control_while` (value-preserving:
// the args feed in, the final carry feeds out, so the rewrite is a plain
// replaceOp). IR-only — the Decision #19 hardware-free Target IR layer. The
// recorded `symbol` (`tessera_apple_gpu_run_graph_while_f32`) is the runtime
// capability that executes the loop (MPSGraph forLoop + select-masking; native
// `while` is unstable). Exercised via the GraphFn lane + the MLIR-driven
// `execute_control_while_mlir` executor.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

using namespace ::mlir;

namespace tessera {
namespace apple {
namespace {

constexpr llvm::StringLiteral kControlWhileOp = "tessera_apple.gpu.control_while";
constexpr llvm::StringLiteral kRunGraphWhileSymbol =
    "tessera_apple_gpu_run_graph_while_f32";

struct LowerControlWhileToAppleGPUPass
    : public PassWrapper<LowerControlWhileToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerControlWhileToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-control-while-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.control_while (bounded while) to the Apple GPU "
           "tessera_apple.gpu.control_while Target-IR op (Phase-G close-out D)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  // Manual module walk (not the greedy driver) — see ControlForToAppleGPU.cpp.
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::SmallVector<Operation *> targets;
    int64_t ordinal = 0;
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.control_while")
        targets.push_back(op);
      else if (name == kControlWhileOp)
        ++ordinal;
    });
    OpBuilder builder(&getContext());
    for (Operation *op : targets) {
      auto body = op->getAttrOfType<FlatSymbolRefAttr>("body");
      auto cond = op->getAttrOfType<FlatSymbolRefAttr>("cond");
      auto carry = op->getAttrOfType<IntegerAttr>("carry_arg_index");
      auto maxIters = op->getAttrOfType<IntegerAttr>("max_iters");
      if (!body || !cond || !carry || !maxIters) {
        op->emitError("tessera.control_while needs body/cond/carry_arg_index/"
                      "max_iters attributes");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(op);
      OperationState st(op->getLoc(), kControlWhileOp);
      st.addOperands(op->getOperands());
      st.addTypes(op->getResultTypes());
      st.addAttribute("body", body);
      st.addAttribute("cond", cond);
      st.addAttribute("carry_arg_index", carry);
      st.addAttribute("max_iters", maxIters);
      st.addAttribute("ordinal", builder.getI64IntegerAttr(ordinal++));
      st.addAttribute("status", builder.getStringAttr("artifact"));
      st.addAttribute("symbol", builder.getStringAttr(kRunGraphWhileSymbol));
      st.addAttribute("framework", builder.getStringAttr("MPSGraph"));
      // Carry the executable body/cond op-list payload through unchanged.
      for (llvm::StringRef k :
           {"body_opcodes", "body_in0", "body_in1", "body_iattr", "body_fattr",
            "body_out_id", "cond_opcodes", "cond_in0", "cond_in1", "cond_iattr",
            "cond_fattr", "cond_out_id"})
        if (Attribute a = op->getAttr(k))
          st.addAttribute(k, a);
      Operation *w = builder.create(st);
      op->replaceAllUsesWith(w);
      op->erase();
    }
  }
};

static PassRegistration<LowerControlWhileToAppleGPUPass> gReg;

} // namespace

std::unique_ptr<Pass> createLowerControlWhileToAppleGPUPass() {
  return std::make_unique<LowerControlWhileToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
