#include "Tessera/Target/Metalium/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace mlir { namespace tessera { namespace metalium {

// Forward decl (implemented in TileToMetalium.cpp)
std::unique_ptr<mlir::Pass> createLowerTileToMetaliumPass();

void registerMetaliumPasses(mlir::DialectRegistry &registry) {
  // If you have dialects to load here, add them; for now this is a stub.
}

// Provide a pipeline like: -pass-pipeline="tessera-metalium(tile-to-metalium)"
static PassPipelineRegistration<> pipeline(
  "tessera-metalium",
  "Lower Tessera Tile IR to Metalium target ops",
  [](OpPassManager &pm) {
    pm.addPass(createLowerTileToMetaliumPass());
  });

}}} // namespace mlir::tessera::metalium
