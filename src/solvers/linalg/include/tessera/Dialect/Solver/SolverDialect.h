#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TesseraLinalgSolverDialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "TesseraLinalgSolverAttrs.h.inc"
#define GET_OP_CLASSES
#include "TesseraLinalgSolverOps.h.inc"

namespace tessera {
namespace solver {

void registerTesseraLinalgSolverDialect(mlir::DialectRegistry &registry);

} // namespace solver
} // namespace tessera
