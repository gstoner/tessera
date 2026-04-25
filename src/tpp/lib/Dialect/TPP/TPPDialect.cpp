
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "TPPDialect.h.inc"

using namespace mlir;
namespace tessera { namespace tpp {

struct TPPDialectImpl {};

class TPPDialect : public Dialect {
public:
  explicit TPPDialect(MLIRContext *ctx)
      : Dialect("tpp", ctx, TypeID::get<TPPDialect>()) {
    addTypes<
      // declared via TPPTypes.cpp.inc registration
    >();
    addAttributes<
      // declared via TPPAttrs.cpp.inc registration
    >();
    addOperations<
      // declared via TPPOps.cpp.inc registration
    >();
  }
};

}} // namespace

#include "TPPDialect.cpp.inc"

static DialectRegistration<tessera::tpp::TPPDialect> TPPReg;
