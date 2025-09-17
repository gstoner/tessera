
//===- tessera_atlas.cpp ---------------------------------------*- C++ -*-===//
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace tessera { namespace atlas {

struct AtlasDialect : public mlir::Dialect {
  explicit AtlasDialect(mlir::MLIRContext *ctx) : Dialect("atlas", ctx) {
    // Register types/ops in real impl; this is a stub for drop-in compilation.
  }
};

}} // namespace tessera::atlas
