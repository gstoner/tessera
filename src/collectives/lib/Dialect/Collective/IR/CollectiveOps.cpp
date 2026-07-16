#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "CollectiveTypes.h.inc"

#define GET_OP_CLASSES
#include "CollectiveOps.h.inc"

#define GET_OP_CLASSES
#include "CollectiveOps.cpp.inc"
