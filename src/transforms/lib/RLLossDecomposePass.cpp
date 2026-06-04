//===- RLLossDecomposePass.cpp - RL policy-loss visibility -----*- C++ -*-===//
//
// Stages 13-15: make PPO/GRPO/CISPO policy losses visible to the compiler.
// PPO's supported envelope is recorded as a primitive formula; GRPO/CISPO are
// now recorded as decomposed reference formulas (normalize/group advantages +
// policy-loss formula) while remaining non-hardware-executable.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

bool isSupportedPPOEnvelope(Operation *op) {
  if (op->getNumOperands() < 3 || op->getNumOperands() > 6 ||
      op->getNumResults() != 1)
    return false;
  auto nextTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto oldTy = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto advTy = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!nextTy || !oldTy || !advTy || !resTy)
    return false;
  if (nextTy.getElementType() != oldTy.getElementType() ||
      nextTy.getElementType() != advTy.getElementType() ||
      nextTy.getElementType() != resTy.getElementType())
    return false;
  if (nextTy.getRank() != oldTy.getRank() || nextTy.getRank() != advTy.getRank())
    return false;
  auto agreeShape = [](RankedTensorType a, RankedTensorType b) {
    for (int64_t i = 0, e = a.getRank(); i < e; ++i) {
      int64_t da = a.getDimSize(i), db = b.getDimSize(i);
      if (!ShapedType::isDynamic(da) && !ShapedType::isDynamic(db) && da != db)
        return false;
    }
    return true;
  };
  if (!agreeShape(nextTy, oldTy) || !agreeShape(nextTy, advTy))
    return false;
  for (Value side : op->getOperands().drop_front(3)) {
    auto sideTy = dyn_cast<RankedTensorType>(side.getType());
    if (!sideTy || sideTy.getElementType() != nextTy.getElementType() ||
        sideTy.getRank() != nextTy.getRank() || !agreeShape(nextTy, sideTy))
      return false;
  }
  if (auto reduction = op->getAttrOfType<StringAttr>("reduction");
      reduction && reduction.getValue() != "mean")
    return false;
  if (auto kl = op->getAttrOfType<FloatAttr>("kl_coef");
      kl && kl.getValueAsDouble() < 0.0)
    return false;
  if (auto entropy = op->getAttrOfType<FloatAttr>("entropy_coef");
      entropy && entropy.getValueAsDouble() < 0.0)
    return false;
  return resTy.getRank() == 0;
}

class RLLossDecomposePass
    : public PassWrapper<RLLossDecomposePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RLLossDecomposePass)

  StringRef getArgument() const override { return "tessera-rl-loss-decompose"; }
  StringRef getDescription() const override {
    return "Mark RL policy-loss ops as compiler-visible and record PPO's "
           "supported primitive-form decomposition envelope";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.rl.ppo_policy_loss") {
        op->setAttr("tessera.rl.compiler_visible", b.getBoolAttr(true));
        op->setAttr("tessera.rl.family", b.getStringAttr("policy_loss"));
        op->setAttr("tessera.rl.variant", b.getStringAttr("ppo"));
        if (isSupportedPPOEnvelope(op)) {
          op->setAttr("tessera.rl.decomposition",
                      b.getStringAttr(
                          "mean(-min(exp(logp_new-logp_old)*adv, "
                          "clip(exp(logp_new-logp_old))*adv))"));
          op->setAttr("tessera.rl.compiler_decomposed_reference",
                      b.getBoolAttr(true));
        } else {
          op->setAttr("tessera.rl.decomposition_status",
                      b.getStringAttr("gated"));
          op->setAttr("tessera.rl.decomposition_reason",
                      b.getStringAttr(
                          "PPO compiler decomposition supports 3 operands, "
                          "mean reduction, matching shapes, and no KL side "
                          "term in Stage 13"));
        }
        return;
      }
      if (name == "tessera.rl.grpo_policy_loss" ||
          name == "tessera.rl.cispo_policy_loss") {
        bool grpo = name == "tessera.rl.grpo_policy_loss";
        op->setAttr("tessera.rl.compiler_visible", b.getBoolAttr(true));
        op->setAttr("tessera.rl.family", b.getStringAttr("policy_loss"));
        op->setAttr("tessera.rl.variant",
                    b.getStringAttr(grpo ? "grpo" : "cispo"));
        op->setAttr("tessera.rl.compiler_decomposed_reference",
                    b.getBoolAttr(true));
        op->setAttr("tessera.rl.decomposition_status",
                    b.getStringAttr("compiler_decomposed_reference"));
        op->setAttr("tessera.rl.decomposition",
                    b.getStringAttr(
                        grpo ? "normalize_group_advantages(rewards) -> "
                               "ppo_policy_loss(logp_new, logp_old, adv)"
                             : "normalize_group_advantages(rewards) -> "
                               "-min(exp(delta), epsilon_high)*adv*logp_new"));
        op->setAttr("tessera.rl.executor_status",
                    b.getStringAttr("non_executable_reference"));
      }
    });
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createRLLossDecomposePass() {
  return std::make_unique<RLLossDecomposePass>();
}
} // namespace tessera
