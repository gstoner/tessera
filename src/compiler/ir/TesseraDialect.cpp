#include "Tessera/IR/Dialects.h"
#include "Tessera/IR/TesseraOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace tessera {

void TesseraDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TesseraOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TesseraOps.cpp.inc"
      >();
}

void registerTesseraDialects(DialectRegistry &registry) {
  registry.insert<TesseraDialect>();
}

#include "TesseraOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "TesseraOpsTypes.cpp.inc"

} // namespace tessera
