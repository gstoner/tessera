#include "tessera/Dialect/Solver/SolverDialect.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace tessera {
namespace solver {

void Tessera_Solver_Dialect::initialize() {
  addAttributes<PrecisionPolicyAttr>();
  addOperations<
#define GET_OP_LIST
#include "TesseraLinalgSolverOps.cpp.inc"
      >();
}

void registerTesseraLinalgSolverDialect(DialectRegistry &registry) {
  registry.insert<Tessera_Solver_Dialect>();
}

LogicalResult GetrfOp::verify() { return success(); }
LogicalResult PotrfOp::verify() { return success(); }

#define GET_ATTRDEF_CLASSES
#include "TesseraLinalgSolverAttrs.cpp.inc"
#define GET_OP_CLASSES
#include "TesseraLinalgSolverOps.cpp.inc"
#include "TesseraLinalgSolverDialect.cpp.inc"

} // namespace solver
} // namespace tessera
