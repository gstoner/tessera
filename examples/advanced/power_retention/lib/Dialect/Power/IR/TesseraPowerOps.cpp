#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "tessera/power/TesseraPowerDialect.h.inc"
#define GET_OP_CLASSES
#include "tessera/power/TesseraPowerOps.h.inc"
#define GET_OP_CLASSES
#include "tessera/power/TesseraPowerOps.cpp.inc"
