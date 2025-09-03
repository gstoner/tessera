#ifndef TESSERA_TARGET_METALIUM_PASSES_H
#define TESSERA_TARGET_METALIUM_PASSES_H

#include <memory>

namespace mlir {
class Pass;
class DialectRegistry;
}

namespace mlir {
namespace tessera {
namespace metalium {

/// Creates the Tile→Metalium lowering pass.
std::unique_ptr<mlir::Pass> createLowerTileToMetaliumPass();

/// Optionally registers all Metalium passes in a registry.
void registerMetaliumPasses(mlir::DialectRegistry &registry);

} // namespace metalium
} // namespace tessera
} // namespace mlir

#endif // TESSERA_TARGET_METALIUM_PASSES_H
