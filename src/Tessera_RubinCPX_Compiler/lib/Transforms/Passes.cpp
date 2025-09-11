
#include "tessera/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

namespace tessera {
void registerCPXPassPipelines() {
  // Example: mlir-opt style pipeline aliases could be added here if desired.
  // For now, rely on individual pass registrations via create*Pass functions.
}
} // namespace tessera
