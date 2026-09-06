// Bounded dense f32 attention products use the registered checkpoint dialect.
// Cache, dropout, bias and numeric-policy variants need their own product ABI.
#ifndef TESSERA_ATTENTION_AD_CONTRACT_H
#define TESSERA_ATTENTION_AD_CONTRACT_H
#include <cmath>
namespace tessera {
inline bool denseAttentionAD(mlir::Operation *op) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1 ||
      op->hasAttr("numeric_policy")) return false;
  for (auto attr : op->getAttrs())
    if (attr.getName() != "head_dim" && attr.getName() != "dropout_p" &&
        attr.getName() != "causal" && attr.getName() != "operandSegmentSizes" &&
        attr.getName() != "tessera.autodiff.activity" && attr.getName() != "tessera.autodiff.role" &&
        attr.getName() != "tessera.effect_kind") return false;
  auto dropout = op->getAttrOfType<mlir::FloatAttr>("dropout_p");
  if (dropout && dropout.getValueAsDouble() != 0.0) return false;
  auto q = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto k = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto v = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(2).getType());
  auto o = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  for (auto t : {q,k,v,o})
    if (!t || t.getRank() != 4 || !t.hasStaticShape() || !t.getElementType().isF32() ||
        llvm::any_of(t.getShape(), [](int64_t d) { return d <= 0; })) return false;
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
  state.addAttribute("scale", b.getF32FloatAttr(1.0 / std::sqrt(double(q.getDimSize(3)))));
  auto causal = source->getAttrOfType<mlir::BoolAttr>("causal");
  state.addAttribute("causal", causal ? causal : b.getBoolAttr(false));
  return b.create(state);
}
}
#endif
