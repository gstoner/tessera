
//===- ResolveTesseraAttrsPass.h - Placeholder Resolver ----------*- C++ -*-===//
// Resolves string-based placeholder attributes (e.g., "tessera.BLOCK_M") used
// in transform dialect scripts into concrete integer attributes so the pipeline
// can run as-is.
//----------------------------------------------------------------------------//
#ifndef TESSERA_RESOLVE_TESSERA_ATTRS_PASS_H
#define TESSERA_RESOLVE_TESSERA_ATTRS_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createResolveTesseraAttrsPass();
} // namespace mlir

#endif // TESSERA_RESOLVE_TESSERA_ATTRS_PASS_H
