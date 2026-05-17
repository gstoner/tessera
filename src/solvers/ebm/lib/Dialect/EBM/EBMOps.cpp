//===- EBMOps.cpp --------------------------------------------*- C++ -*-===//
#include "tessera/EBM/EBMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace tessera::ebm;

#define GET_OP_CLASSES
#include "EBMOps.cpp.inc"
