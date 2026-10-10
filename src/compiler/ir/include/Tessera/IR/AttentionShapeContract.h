#ifndef TESSERA_IR_ATTENTION_SHAPE_CONTRACT_H
#define TESSERA_IR_ATTENTION_SHAPE_CONTRACT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>

// Shared semantic capacity resolution for native attention products.
// Symbolic sequence axes survive into Schedule; capacities never relabel them.
// Shared by the semantic dialect verifier and native Schedule projection.
namespace tessera {
struct NativeAttentionShape {
  mlir::DenseI64ArrayAttr bounds;
  bool dynamicSequence;
};
inline mlir::FailureOr<NativeAttentionShape> resolveNativeAttentionShape(
    mlir::Operation *op, llvm::ArrayRef<int64_t> symbolic) {
  using namespace mlir;
  if (symbolic.size() != 7)
    return op->emitError("attention shape requires seven logical dimensions"), failure();
  auto mod = op->getParentOfType<ModuleOp>();
  auto boundsRaw = mod ? mod->getAttr("tessera.attention_shape_bounds") : Attribute();
  auto bounds = dyn_cast_or_null<DenseI64ArrayAttr>(boundsRaw);
  bool dynamicSequence = ShapedType::isDynamic(symbolic[3]) || ShapedType::isDynamic(symbolic[4]);
  if (boundsRaw && (!bounds || bounds.size() != 7))
    return op->emitError("checkpoint sequence bounds require seven i64 capacities"), failure();
  if (dynamicSequence != bool(bounds))
    return op->emitError("dynamic checkpoint sequences require explicit native shape bounds"), failure();
  for (unsigned axis = 0; axis < symbolic.size(); ++axis) {
    bool dynamic = ShapedType::isDynamic(symbolic[axis]);
    if ((dynamic && axis != 3 && axis != 4) || (!dynamic && symbolic[axis] <= 0))
      return op->emitError("checkpoint only sequence axes may be dynamic"), failure();
    if (bounds && (bounds[axis] <= 0 || (!dynamic && bounds[axis] != symbolic[axis])))
      return op->emitError("checkpoint capacity must preserve fixed dimensions"), failure();
  }
  if (bounds) {
    // Reject capacity products that overflow the checked byte-address ABI.
    // Check each physical tensor, rather than multiplying unrelated roles.
    for (SmallVector<unsigned> axes : {SmallVector<unsigned>{0,1,3,5},
                                      SmallVector<unsigned>{0,2,4,5},
                                      SmallVector<unsigned>{0,2,4,6},
                                      SmallVector<unsigned>{0,1,3,6},
                                      SmallVector<unsigned>{0,1,3,4}}) {
      int64_t capacity = 4;
      for (unsigned axis : axes) {
        int64_t extent = bounds[axis];
        if (extent > std::numeric_limits<int64_t>::max() / capacity)
        return op->emitError("checkpoint capacity exceeds the byte-address ABI"), failure();
        capacity *= extent;
      }
    }
  }

  return NativeAttentionShape{bounds, dynamicSequence};
}
} // namespace tessera

#endif
