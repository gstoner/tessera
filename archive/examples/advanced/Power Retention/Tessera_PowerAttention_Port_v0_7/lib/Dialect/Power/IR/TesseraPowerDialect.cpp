#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "tessera/power/TesseraPowerDialect.h.inc"
using namespace mlir;
namespace tessera { namespace power {
struct PowerDialect : public ::mlir::Dialect {
  explicit PowerDialect(MLIRContext *ctx) : Dialect(getDialectNamespace(), ctx, TypeID::get<PowerDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "tessera/power/TesseraPowerOps.cpp.inc"
    >();
  }
  static StringRef getDialectNamespace() { return "power"; }
};
}} // ns
#include "tessera/power/TesseraPowerDialect.cpp.inc"
