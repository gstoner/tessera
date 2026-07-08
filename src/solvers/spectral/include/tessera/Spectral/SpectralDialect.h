//===- SpectralDialect.h ---------------------------------------*- C++ -*-===//
#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"    // BytecodeOpInterface
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"                  // mlir::Op, getProperties
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"  // Pure trait

#include "SpectralDialect.h.inc" // generated dialect decl

// Generated type declarations (e.g. !tessera_spectral.plan).  Op class
// declarations/definitions are pulled in at global scope inside the dialect
// TUs (SpectralDialect.cpp / SpectralOps.cpp); the passes identify ops by
// name and do not need the generated op classes.
#define GET_TYPEDEF_CLASSES
#include "SpectralTypes.h.inc"
