
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>
namespace tessera {
std::unique_ptr<mlir::Pass> createPartitionLongContextPass();
std::unique_ptr<mlir::Pass> createLowerKVTransportPass();
std::unique_ptr<mlir::Pass> createNVFP4VectorizePass();
std::unique_ptr<mlir::Pass> createFuseVideoIngestPass();
void registerCPXPasses();
void registerCPXPipeline();
} // namespace tessera
