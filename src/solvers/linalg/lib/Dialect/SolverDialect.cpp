#include "tessera/Dialect/Solver/SolverDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace tessera {
namespace solver {

void TesseraSolverDialect::initialize() {
  addAttributes<PrecisionPolicyAttr>();
  addOperations<
#define GET_OP_LIST
#include "TesseraLinalgSolverOps.cpp.inc"
      >();
}

void registerTesseraLinalgSolverDialect(DialectRegistry &registry) {
  registry.insert<TesseraSolverDialect>();
}

LogicalResult GetrfOp::verify() { return success(); }
LogicalResult PotrfOp::verify() { return success(); }

} // namespace solver
} // namespace tessera

#define GET_ATTRDEF_CLASSES
#include "TesseraLinalgSolverAttrs.cpp.inc"
#define GET_OP_CLASSES
#include "TesseraLinalgSolverOps.cpp.inc"
#include "TesseraLinalgSolverDialect.cpp.inc"
