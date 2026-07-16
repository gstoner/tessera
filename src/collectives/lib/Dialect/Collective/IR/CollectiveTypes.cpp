#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "CollectiveTypes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "CollectiveTypes.cpp.inc"
