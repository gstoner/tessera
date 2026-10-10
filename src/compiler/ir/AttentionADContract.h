// Bounded dense f32 attention products use the registered checkpoint dialect.
// Forward and reverse products admit rank-four f32 score biases broadcast on any axis.
// JVP products carry bias and its tangent explicitly; cache, dropout and
// numeric-policy variants require separate contracts.
#ifndef TESSERA_ATTENTION_AD_CONTRACT_H
#define TESSERA_ATTENTION_AD_CONTRACT_H
#include <cmath>
#include "mlir/IR/BuiltinOps.h"
#include "Tessera/IR/AttentionShapeContract.h"
namespace tessera {
inline bool denseAttentionAD(mlir::Operation *op, bool allowBias = false, bool allowLse = false, bool allowBoundedSequences = false) {
  if ((op->getNumOperands() != 3 && !(allowBias && op->getNumOperands() == 4)) || (op->getNumResults() != 1 && !(allowLse && op->getNumResults() == 2)) ||
      op->hasAttr("numeric_policy")) return false;
  for (auto attr : op->getAttrs())
    if (attr.getName() != "head_dim" && attr.getName() != "dropout_p" &&
        attr.getName() != "causal" && attr.getName() != "operandSegmentSizes" &&
        !(allowLse && op->getNumResults() == 2 && attr.getName() == "lse_checkpoint") &&
        attr.getName() != "tessera.autodiff.activity" && attr.getName() != "tessera.autodiff.role" &&
        attr.getName() != "tessera.effect_kind") return false;
  auto dropout = op->getAttrOfType<mlir::FloatAttr>("dropout_p");
  if (dropout && dropout.getValueAsDouble() != 0.0) return false;
  auto q = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto k = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto v = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(2).getType());
  auto o = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  for (auto t : {q,k,v,o})
    if (!t || t.getRank() != 4 || !t.getElementType().isF32() ||
        llvm::any_of(t.getShape(), [](int64_t d) { return !mlir::ShapedType::isDynamic(d) && d <= 0; })) return false;
  bool dynamic = !q.hasStaticShape() || !k.hasStaticShape() || !v.hasStaticShape() || !o.hasStaticShape();
  if (dynamic) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto target = module ? module->getAttrOfType<mlir::StringAttr>("tessera.target") : mlir::StringAttr();
    auto arch = module ? module->getAttrOfType<mlir::StringAttr>("tessera.arch") : mlir::StringAttr();
    auto bounds = module ? module->getAttrOfType<mlir::DenseI64ArrayAttr>("tessera.attention_shape_bounds") : mlir::DenseI64ArrayAttr();
    if (!allowBoundedSequences || !target || target.getValue()!="nvidia_sm120" ||
        !arch || arch.getValue()!="sm_120" || !bounds || bounds.size()!=7) return false;
    llvm::SmallVector<int64_t> dims{q.getDimSize(0),q.getDimSize(1),k.getDimSize(1),
        q.getDimSize(2),k.getDimSize(2),q.getDimSize(3),v.getDimSize(3)};
    auto shape = resolveNativeAttentionShape(op, dims);
    if (mlir::failed(shape)) return false;
  }
  if (q.getDimSize(0)<=0 || q.getDimSize(1)<=0 || k.getDimSize(1)<=0 ||
      q.getDimSize(3)<=0 || v.getDimSize(3)<=0) return false;
  if (op->getNumResults() == 2) {
    auto policy = op->getAttrOfType<mlir::StringAttr>("lse_checkpoint");
    auto lse = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
    if (!policy || policy.getValue() != "saved" || !lse ||
        lse != mlir::RankedTensorType::get(q.getShape().take_front(3), q.getElementType())) return false;
  }
  if (op->getNumOperands() == 4) {
    auto bias = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(3).getType());
    if (!bias || bias.getRank() != 4 || !bias.getElementType().isF32() ||
        (!bias.hasStaticShape() && !dynamic)) return false;
    llvm::SmallVector<int64_t, 4> scores{q.getDimSize(0), q.getDimSize(1),
                                        q.getDimSize(2), k.getDimSize(2)};
    for (unsigned axis = 0; axis < 4; ++axis)
      if (bias.getDimSize(axis) != 1 && bias.getDimSize(axis) != scores[axis])
        return false;
  }
  auto head = op->getAttrOfType<mlir::IntegerAttr>("head_dim");
  if (!head || head.getInt() != q.getDimSize(3)) return false;
  return q.getDimSize(0) == k.getDimSize(0) && k.getDimSize(0) == v.getDimSize(0) &&
    q.getDimSize(1) % k.getDimSize(1) == 0 && k.getDimSize(1) == v.getDimSize(1) &&
    k.getDimSize(2) == v.getDimSize(2) && q.getDimSize(3) == k.getDimSize(3) &&
    o.getShape() == llvm::ArrayRef<int64_t>({q.getDimSize(0), q.getDimSize(1), q.getDimSize(2), v.getDimSize(3)});
}
inline mlir::Operation *attentionCheckpoint(mlir::OpBuilder &b, mlir::Operation *source,
                                            bool backward, mlir::ValueRange operands) {
  if (!b.getContext()->getOrLoadDialect("tessera_attn")) return nullptr;
  auto q = mlir::cast<mlir::RankedTensorType>(source->getOperand(0).getType());
  mlir::OperationState state(source->getLoc(), backward ? "tessera_attn.checkpoint_backward" : "tessera_attn.checkpoint_forward");
  state.addOperands(operands);
  if (backward) for (auto value : source->getOperands()) state.addTypes(value.getType());
  else {
    state.addTypes(source->getResult(0).getType());
    state.addTypes(mlir::RankedTensorType::get(q.getShape().take_front(3), b.getF32Type()));
  }
  if (backward && source->getNumResults() == 2)
    state.addAttribute("lse_cotangent", b.getBoolAttr(true));
  state.addAttribute("scale", b.getF32FloatAttr(1.0 / std::sqrt(double(q.getDimSize(3)))));
  auto causal = source->getAttrOfType<mlir::BoolAttr>("causal");
  state.addAttribute("causal", causal ? causal : b.getBoolAttr(false));
  return b.create(state);
}
}
#endif
