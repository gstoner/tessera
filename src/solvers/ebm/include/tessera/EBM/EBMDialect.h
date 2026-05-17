//===- EBMDialect.h ---------------------------------------------*- C++ -*-===//
#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "EBMDialect.h.inc"  // generated decls

#define GET_OP_CLASSES
#include "EBMOps.h.inc"
