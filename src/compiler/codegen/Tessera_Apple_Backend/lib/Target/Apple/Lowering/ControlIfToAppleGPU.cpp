//===- ControlIfToAppleGPU.cpp - Lower tessera.control_if --------------===//
//
// Phase-G close-out C — lower the Graph-IR divergent if/else `tessera.control_if`
// to the Apple Target-IR op `tessera_apple.gpu.control_if` (value-preserving: the
// args feed in, the selected branch result feeds out, so the rewrite is a plain
// replaceOp). IR-only — the Decision #19 hardware-free Target IR layer. The
// recorded `symbol` (`tessera_apple_gpu_run_graph_cond_f32`) is the runtime
// capability that executes the if (exercised via the GraphFn lane + the
// MLIR-driven `execute_control_if_mlir` executor).
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

constexpr llvm::StringLiteral kControlIfOp = "tessera_apple.gpu.control_if";
constexpr llvm::StringLiteral kRunGraphCondSymbol =
    "tessera_apple_gpu_run_graph_cond_f32";

struct LowerControlIfToAppleGPUPass
    : public PassWrapper<LowerControlIfToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerControlIfToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-control-if-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower tessera.control_if (divergent if/else) to the Apple GPU "
           "tessera_apple.gpu.control_if Target-IR op (Phase-G close-out C)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  // Manual module walk (not the greedy driver) — see ControlForToAppleGPU.cpp:
  // the greedy driver's region simplification would DCE unrelated result-less
  // artifact ops before sibling passes lower them. This pass only touches
  // `tessera.control_if` ops.
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::SmallVector<Operation *> targets;
    int64_t ordinal = 0;
    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.control_if")
        targets.push_back(op);
      else if (name == kControlIfOp)
        ++ordinal;
    });
    OpBuilder builder(&getContext());
    for (Operation *op : targets) {
      auto thenB = op->getAttrOfType<FlatSymbolRefAttr>("then_branch");
      auto elseB = op->getAttrOfType<FlatSymbolRefAttr>("else_branch");
      auto flag = op->getAttrOfType<IntegerAttr>("flag_arg_index");
      if (!thenB || !elseB || !flag) {
        op->emitError("tessera.control_if needs then_branch/else_branch/"
                      "flag_arg_index attributes");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(op);
      OperationState st(op->getLoc(), kControlIfOp);
      st.addOperands(op->getOperands());
      st.addTypes(op->getResultTypes());
      st.addAttribute("then_branch", thenB);
      st.addAttribute("else_branch", elseB);
      st.addAttribute("flag_arg_index", flag);
      st.addAttribute("ordinal", builder.getI64IntegerAttr(ordinal++));
      st.addAttribute("status", builder.getStringAttr("artifact"));
      st.addAttribute("symbol", builder.getStringAttr(kRunGraphCondSymbol));
      st.addAttribute("framework", builder.getStringAttr("MPSGraph"));
      // Carry the executable then/else op-list payload through unchanged.
      for (llvm::StringRef k :
           {"then_opcodes", "then_in0", "then_in1", "then_iattr", "then_fattr",
            "then_out_id", "else_opcodes", "else_in0", "else_in1", "else_iattr",
            "else_fattr", "else_out_id", "out_shape"})
        if (Attribute a = op->getAttr(k))
          st.addAttribute(k, a);
      Operation *ifOp = builder.create(st);
      op->replaceAllUsesWith(ifOp);
      op->erase();
    }
  }
};

static PassRegistration<LowerControlIfToAppleGPUPass> gReg;

} // namespace

std::unique_ptr<Pass> createLowerControlIfToAppleGPUPass() {
  return std::make_unique<LowerControlIfToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
