
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TPPDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "TPPTypes.h.inc"
#define GET_ATTRDEF_CLASSES
#include "TPPAttrs.h.inc"
#define GET_OP_CLASSES
#include "TPPOps.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TPPTypes.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "TPPAttrs.cpp.inc"
#define GET_OP_CLASSES
#include "TPPOps.cpp.inc"

namespace tessera {
namespace tpp {

void TPPDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TPPTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TPPAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TPPOps.cpp.inc"
      >();
}

} // namespace tpp
} // namespace tessera

#include "TPPDialect.cpp.inc"
