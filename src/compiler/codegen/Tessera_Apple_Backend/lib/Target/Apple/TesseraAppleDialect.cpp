//===- TesseraAppleDialect.cpp - Apple Silicon Target IR ------*- C++ -*-===//
//
// Dialect / op registration for the hardware-free Apple Silicon Target IR.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/IR/DialectImplementation.h"

#include "Tessera/Target/Apple/TesseraAppleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Tessera/Target/Apple/TesseraAppleOps.cpp.inc"

namespace tessera {
namespace apple {

void TesseraApple_Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tessera/Target/Apple/TesseraAppleOps.cpp.inc"
      >();
}

void registerAppleDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraApple_Dialect>();
}

} // namespace apple
} // namespace tessera
