#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/Pass/Pass.h"

namespace tessera {
void registerTesseraTPUPasses() {
  ::mlir::PassRegistration<>(createLowerTesseraToStableHLOPass);
  ::mlir::PassRegistration<>(createAnnotateShardingPass);
}
} // namespace tessera
