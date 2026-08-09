#pragma once

#include <memory>

namespace mlir {
class DialectRegistry;
class Pass;
} // namespace mlir

namespace tessera::collective {

std::unique_ptr<mlir::Pass> createLowerTileCollectivesPass();
void registerCollectivePasses();

} // namespace tessera::collective
