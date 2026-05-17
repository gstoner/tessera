//===- EBMDialect.cpp ----------------------------------------*- C++ -*-===//
#include "tessera/EBM/EBMDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace tessera::ebm;

#include "EBMDialect.cpp.inc"

void EBMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "EBMOps.cpp.inc"
      >();
}
