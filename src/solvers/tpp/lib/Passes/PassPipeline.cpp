//===- PassPipeline.cpp - Legacy entry point for the TPP space-time alias -===//
//
// The canonical entry points now live in lib/InitTPP.cpp:
// `tessera::tpp::registerTPPPipelines()` registers the `-tpp-space-time`
// alias.  This file preserves the legacy `extern "C"` shim
// (`registerTPPPipelineAlias`) so any out-of-tree caller built against
// an older TPP header keeps working — it just delegates to the new
// entry point.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "tpp/InitTPP.h"

extern "C" void registerTPPPipelineAlias(mlir::PassPipelineRegistration<>*) {
  ::tessera::tpp::registerTPPPipelines();
}
