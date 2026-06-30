// ControlFlowTargetGuardPass.cpp — CF0 unsupported-control-flow diagnostic
//
// CF0 of docs/audit/roadmap/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md and
// the contract in docs/spec/CONTROL_FLOW_CONTRACT.md §5.
//
// The Graph IR control-flow ops (`tessera.control_for` / `control_if` /
// `control_while` / `control_scan`) have target-specific executable envelopes.
// Apple GPU has its Target-IR path; ROCm CF4 covers a narrow elementwise rank-1
// control_for/if/while subset. Forms outside a backend's envelope still need a
// stable diagnostic instead of falling through to a confusing downstream failure
// or a silent host-loop fallback inside an "executable backend" claim.
//
// Per Decision #21 (unsupported lowering must emit a STABLE diagnostic naming
// the op and the target, never silently no-op), this guard walks for the four
// control ops and fails loudly with a fixed diagnostic code. It is wired into
// every non-Apple lowering pipeline; the `target` option only names the backend
// in the message (detection is target-independent). Backends with partial
// control-flow support should run their envelope-specific lowering before this
// guard so only leftover unsupported forms are rejected.

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
    return "CF0: reject tessera.control_{for,if,while,scan} forms that remain "
           "outside the selected backend's control-flow envelope with a stable "
           "CONTROL_FLOW_UNSUPPORTED_ON_TARGET diagnostic (Decision #21).";
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
          << "' for this control-flow form/envelope; use a target-supported "
             "subset or hoist the loop/branch to the host "
             "(see docs/spec/CONTROL_FLOW_CONTRACT.md).";
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
