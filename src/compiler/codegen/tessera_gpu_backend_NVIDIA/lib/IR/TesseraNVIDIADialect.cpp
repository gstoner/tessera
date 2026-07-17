#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TesseraNVIDIADialect.h.inc"

#define GET_OP_CLASSES
#include "TesseraNVIDIAOps.h.inc"

using namespace mlir;
using namespace tessera::nvidia;

#include "TesseraNVIDIADialect.cpp.inc"

#define GET_OP_CLASSES
#include "TesseraNVIDIAOps.cpp.inc"

void TesseraNVIDIADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TesseraNVIDIAOps.cpp.inc"
      >();
}
