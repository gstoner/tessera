//===- NeighborsDialect.h — Registration for the Neighbors dialect ---------===//
//
// Public header exposing the dialect registration entry point. Used by
// `tessera-opt` and any tool that needs to parse `tessera.neighbors.*` IR.
//
// The dialect itself is defined in lib/Dialect/Neighbors/IR/TesseraNeighbors.cpp.
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir { class DialectRegistry; }

namespace tessera {
namespace neighbors {

/// Registers the `tessera.neighbors` dialect into the given registry.
void registerNeighborsDialect(::mlir::DialectRegistry &registry);

} // namespace neighbors
} // namespace tessera
