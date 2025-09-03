#ifndef TESSERA_TARGET_METALIUM_DIALECT_H
#define TESSERA_TARGET_METALIUM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace tessera {
namespace metalium {

class TesseraMetaliumDialect : public mlir::Dialect {
public:
  explicit TesseraMetaliumDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "tessera_metalium"; }
  void initialize() override;
};

/// Register function for linking into a registry.
void registerMetaliumDialect(mlir::DialectRegistry &registry);

} // namespace metalium
} // namespace tessera
} // namespace mlir

#endif // TESSERA_TARGET_METALIUM_DIALECT_H
