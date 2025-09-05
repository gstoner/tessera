#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "CollectiveOps.h.inc"
using namespace mlir; using namespace tessera::collective;
LogicalResult AwaitOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location>, ValueRange operands,
                                        DictionaryAttr, RegionRange, SmallVectorImpl<Type>&tys){
  if (operands.empty()) return failure();
  auto f = dyn_cast<Tessera_FutureType>(operands.front().getType());
  if (!f) return failure();
  tys.push_back(f.getValueType()); return success();
}
