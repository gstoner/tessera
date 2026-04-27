#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "TesseraOpsEnums.h.inc"
#include "TesseraOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "TesseraOps.h.inc"
