//===- CliffordOps.cpp ----------------------------------------*- C++ -*-===//
#include "tessera/Clifford/CliffordDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace tessera::clifford;

// Generated op-defs must be at file scope (MLIR 23).
#define GET_OP_CLASSES
#include "CliffordOps.cpp.inc"
