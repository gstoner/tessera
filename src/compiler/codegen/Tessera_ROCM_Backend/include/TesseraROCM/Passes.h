#pragma once
#include "mlir/Pass/Pass.h"
namespace mlir {
class Pass;
namespace tessera_rocm {
std::unique_ptr<mlir::Pass> createLowerTesseraTargetToROCDLPass();
std::unique_ptr<mlir::Pass> createLowerKernelABIPass();
void registerTesseraROCMPasses();
}}
