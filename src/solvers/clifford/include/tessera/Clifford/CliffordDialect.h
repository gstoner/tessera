//===- CliffordDialect.h ----------------------------------------*- C++ -*-===//
// Tessera Clifford / Geometric Algebra dialect header.
//
// Generated decls come from `mlir_tablegen ... -gen-dialect-decls`; the
// `.h.inc` file is produced by `CliffordOps.td` in the parent CMakeLists.
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "CliffordDialect.h.inc"  // generated decls

#define GET_OP_CLASSES
#include "CliffordOps.h.inc"  // generated op decls
