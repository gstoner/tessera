// ControlFlowTargetGuardPass.cpp — CF0 unsupported-control-flow diagnostic
//
// CF0 of docs/audit/roadmap/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md and
// the contract in docs/spec/CONTROL_FLOW_CONTRACT.md §5.
//
// The Graph IR control-flow ops (`tessera.control_for` / `control_if` /
// `control_while` / `control_scan`) lower to device code on Apple GPU today
// (ControlForToAppleGPU / ControlWhileToAppleGPU + run_graph_scan_f32). No other
// backend has a control-flow lowering yet — CF3 (CUDA) and CF4 (ROCm) build
// them. Until then, a control-flow program targeting CUDA / ROCm / x86 hit NO
// lowering pattern AND no diagnostic: it would fall through to a confusing
// downstream failure or — worse — a silent host-loop fallback inside an
// "executable backend" claim.
//
// Per Decision #21 (unsupported lowering must emit a STABLE diagnostic naming
// the op and the target, never silently no-op), this guard walks for the four
// control ops and fails loudly with a fixed diagnostic code. It is wired into
// every non-Apple lowering pipeline; the `target` option only names the backend
// in the message (detection is target-independent). When CF3/CF4 land real
// kernels, drop the guard from that backend's pipeline.

#include "Tessera/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

// The Graph IR control-flow op names (matched by string so the guard works on
// both the registered and the generic `"tessera.control_*"(...)` forms).
static bool isControlFlowOp(llvm::StringRef name) {
  return name == "tessera.control_for" || name == "tessera.control_if" ||
         name == "tessera.control_while" || name == "tessera.control_scan";
}

struct ControlFlowTargetGuard
    : public PassWrapper<ControlFlowTargetGuard, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ControlFlowTargetGuard)

  ControlFlowTargetGuard() = default;
  explicit ControlFlowTargetGuard(llvm::StringRef t) { target = t.str(); }
  ControlFlowTargetGuard(const ControlFlowTargetGuard &other)
      : PassWrapper(other) {
    target = other.target;
  }

  // The backend name, used only to make the diagnostic name the target.
  Option<std::string> target{*this, "target",
                             llvm::cl::desc("target backend name for the "
                                            "diagnostic message"),
                             llvm::cl::init("this backend")};

  StringRef getArgument() const override {
    return "tessera-control-flow-target-guard";
  }
  StringRef getDescription() const override {
    return "CF0: reject tessera.control_{for,if,while,scan} on backends without "
           "a control-flow lowering (everything but apple_gpu today) with a "
           "stable CONTROL_FLOW_UNSUPPORTED_ON_TARGET diagnostic (Decision "
           "#21); CF3/CF4 replace it with executable CUDA/ROCm kernels.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;
    module.walk([&](Operation *op) {
      llvm::StringRef name = op->getName().getStringRef();
      if (!isControlFlowOp(name))
        return;
      op->emitOpError("CONTROL_FLOW_UNSUPPORTED_ON_TARGET: '")
          << name << "' is not yet executable on target '" << target
          << "'; device control-flow lowering for this backend lands in "
             "CF3 (CUDA) / CF4 (ROCm). Only apple_gpu lowers control flow "
             "today (see docs/spec/CONTROL_FLOW_CONTRACT.md).";
      anyError = true;
    });
    if (anyError)
      signalPassFailure();
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createControlFlowTargetGuardPass(llvm::StringRef target) {
  return std::make_unique<ControlFlowTargetGuard>(target);
}
}  // namespace tessera
