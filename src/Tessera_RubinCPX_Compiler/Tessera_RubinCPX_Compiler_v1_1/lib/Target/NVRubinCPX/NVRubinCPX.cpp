
#include "tessera/Target/NVRubinCPX/NVRubinCPX.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
namespace tessera { namespace target {
NVRubinCPXDialect::NVRubinCPXDialect(MLIRContext *ctx)
  : Dialect(getDialectNamespace(), ctx, TypeID::get<NVRubinCPXDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "NVRubinCPXOps.cpp.inc"
  >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "NVRubinCPXTypes.cpp.inc"
  >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "NVRubinCPXAttrs.cpp.inc"
  >();
}
}} // namespace

#define GET_OP_CLASSES
#include "NVRubinCPXOps.cpp.inc"
