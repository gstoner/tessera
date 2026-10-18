// Shared semantic projection of an actual generated checkpoint adjoint.
#pragma once
#include "mlir/IR/Builders.h"

namespace tessera::attention_rows {
inline mlir::Operation *createProjection(
    mlir::OpBuilder &builder, mlir::Operation *backward,
    mlir::ValueRange operands, mlir::TypeRange results, bool consumer) {
  auto seeded = backward->getAttrOfType<mlir::BoolAttr>("lse_cotangent");
  bool hasSeed = seeded && seeded.getValue();
  bool bias = backward->getNumOperands() == 7 + unsigned(hasSeed);
  mlir::OperationState state(backward->getLoc(), consumer ?
      "tessera_attn.checkpoint_backward_rows" : "tessera_attn.checkpoint_row_correction");
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttribute("scale", backward->getAttr("scale"));
  state.addAttribute("causal", backward->getAttr("causal"));
  state.addAttribute("bias", builder.getBoolAttr(bias));
  state.addAttribute("lse_cotangent", builder.getBoolAttr(hasSeed));
  return builder.create(state);
}
} // namespace tessera::attention_rows
