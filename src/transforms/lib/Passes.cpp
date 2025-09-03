
#include "Tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
void registerTesseraPasses() {
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-canonicalize",
      "Canonicalize common Tessera IR patterns",
      []() { return createCanonicalizeTesseraIRPass(); });

  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-verify",
      "Verify structural and attribute constraints for key Tessera IR ops",
      []() { return createVerifyTesseraIRPass(); });
}
} // namespace tessera
