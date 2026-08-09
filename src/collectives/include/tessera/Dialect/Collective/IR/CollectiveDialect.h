#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "CollectiveDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "CollectiveTypes.h.inc"
#define GET_OP_CLASSES
#include "CollectiveOps.h.inc"

namespace tessera::collective {

void registerCollectiveDialect(mlir::DialectRegistry &registry);

} // namespace tessera::collective
