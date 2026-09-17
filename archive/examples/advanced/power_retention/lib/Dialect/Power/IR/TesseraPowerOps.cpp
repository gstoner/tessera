#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

#include "tessera/power/TesseraPowerDialect.h.inc"
#define GET_OP_CLASSES
#include "tessera/power/TesseraPowerOps.h.inc"
#define GET_OP_CLASSES
#include "tessera/power/TesseraPowerOps.cpp.inc"
