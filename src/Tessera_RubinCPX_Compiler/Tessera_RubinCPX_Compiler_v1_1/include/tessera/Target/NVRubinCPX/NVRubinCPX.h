
#pragma once
#include "mlir/IR/Dialect.h"
namespace tessera { namespace target {
struct NVRubinCPXDialect : public mlir::Dialect {
  explicit NVRubinCPXDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "tessera.target.cpx"; }
};
}} // namespace
