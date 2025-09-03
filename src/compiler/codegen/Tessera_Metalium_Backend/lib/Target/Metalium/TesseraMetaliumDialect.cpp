#include "Tessera/Target/Metalium/TesseraMetaliumDialect.h"

// Generated op declarations (produced by tblgen from TesseraMetaliumOps.td):
// In a real build, CMake invokes mlir-tblgen to create these headers.
// Here we include them conditionally so the file is self-contained.
#ifdef GEN_TESSERA_METALIUM_OPS_DECL
  #include "Tessera/Target/Metalium/TesseraMetaliumOps.h.inc"
#endif

using namespace mlir;
using namespace mlir::tessera::metalium;

TesseraMetaliumDialect::TesseraMetaliumDialect(MLIRContext *ctx)
  : Dialect(getDialectNamespace(), ctx, TypeID::get<TesseraMetaliumDialect>()) {
  initialize();
}

void TesseraMetaliumDialect::initialize() {
  // Normally we'd add types/attributes here and register ops:
  // addOperations<
  //   Metalium_DmaOp, Metalium_Load2DOp, Metalium_Store2DOp, Metalium_MatmulOp
  // >();
  // For header-only stubbing without tblgen, we keep this as a placeholder.
}

void mlir::tessera::metalium::registerMetaliumDialect(DialectRegistry &registry) {
  registry.insert<TesseraMetaliumDialect>();
}
