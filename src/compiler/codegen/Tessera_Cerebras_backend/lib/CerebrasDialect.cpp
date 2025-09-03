#include "tessera/targets/cerebras/Passes.h"
#if HAVE_MLIR
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "CerebrasDialect.h.inc"
#include "CerebrasOps.h.inc"

namespace tessera { namespace cerebras {
using namespace mlir;

struct CerebrasDialectImpl : public ::mlir::Dialect {
  explicit CerebrasDialectImpl(MLIRContext *ctx)
      : Dialect("cerebras", ctx, TypeID::get<CerebrasDialectImpl>()) {
    addOperations<
#define GET_OP_LIST
#include "CerebrasOps.cpp.inc"
    >();
  }
};

static DialectRegistration<CerebrasDialectImpl> Cerebras;
}} // namespace

#else
// No-MLIR build
#endif
