//===- TesseraX86Ops.cpp — tessera_x86 dialect registration ---------------===//
//
// Built by W0.10 so x86 stops being the one backend that silently ignores
// Decision #19. See TesseraX86Dialect.td for why no carve-out was granted.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TesseraX86Dialect.h"
#define GET_TYPEDEF_CLASSES
#include "TesseraX86Types.h.inc"
#define GET_OP_CLASSES
#include "TesseraX86Ops.h.inc"

using namespace mlir::tessera_x86;

#include "TesseraX86Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TesseraX86Types.cpp.inc"

#define GET_OP_CLASSES
#include "TesseraX86Ops.cpp.inc"

void mlir::tessera_x86::registerTesseraX86Dialect(
    ::mlir::DialectRegistry &registry) {
  registry.insert<TesseraX86Dialect>();
}

void TesseraX86Dialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TesseraX86Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TesseraX86Ops.cpp.inc"
      >();
}
