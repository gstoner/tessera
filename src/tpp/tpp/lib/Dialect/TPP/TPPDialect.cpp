
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace tessera { namespace tpp {
class TPPDialect : public Dialect {
public:
  explicit TPPDialect(MLIRContext *ctx) : Dialect("tpp", ctx, TypeID::get<TPPDialect>()) {
    // Register ops/types later.
  }
};
}} // namespace

static DialectRegistration<tessera::tpp::TPPDialect> TPPReg;
