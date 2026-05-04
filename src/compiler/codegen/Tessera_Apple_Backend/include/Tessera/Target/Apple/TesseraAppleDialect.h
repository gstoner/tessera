//===- TesseraAppleDialect.h - Apple Silicon Target IR --------*- C++ -*-===//
//
// Hardware-free Target IR dialect for Apple Silicon CPU (Accelerate / vecLib /
// BNNS) and GPU (Metal / MPS) artifacts. Mirrors the ROCm and Metalium
// backend pattern (Architecture Decision #19).
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_DIALECT_H
#define TESSERA_TARGET_APPLE_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated dialect declarations (cppNamespace = ::tessera::apple).
#include "Tessera/Target/Apple/TesseraAppleDialect.h.inc"

#define GET_OP_CLASSES
#include "Tessera/Target/Apple/TesseraAppleOps.h.inc"

namespace tessera {
namespace apple {

/// Insert the Apple Target IR dialect into a DialectRegistry. Call from
/// tessera-opt and any other tool that needs to parse / verify Apple IR.
void registerAppleDialect(::mlir::DialectRegistry &registry);

} // namespace apple
} // namespace tessera

#endif // TESSERA_TARGET_APPLE_DIALECT_H
