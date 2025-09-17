#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/MLIRContext.h"

#include "P3DOps.h.inc"

using namespace mlir;

namespace mlir { namespace tessera { namespace p3d {

struct P3DDialect : public Dialect {
  explicit P3DDialect(MLIRContext *ctx) : Dialect(getDialectNamespace(), ctx, TypeID::get<P3DDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "P3DOps.cpp.inc"
    >();
  }
  static StringRef getDialectNamespace() { return "p3d"; }
};

}}} // namespace
