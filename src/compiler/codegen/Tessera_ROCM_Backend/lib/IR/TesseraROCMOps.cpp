#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TesseraROCMDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "TesseraROCMTypes.h.inc"
#define GET_OP_CLASSES
#include "TesseraROCMOps.h.inc"

using namespace mlir::tessera_rocm;

#include "TesseraROCMDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TesseraROCMTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TesseraROCMOps.cpp.inc"

void TesseraROCMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TesseraROCMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TesseraROCMOps.cpp.inc"
  >();
}
