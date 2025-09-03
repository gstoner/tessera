#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tessera_rocm {
struct ABIConfig {
  std::string mcpu = "gfx90a";
  unsigned wgX = 256, wgY = 1, wgZ = 1;
  unsigned ldsBytes = 0;
};
void annotateKernelABI(mlir::func::FuncOp fn, const ABIConfig &cfg);
} // namespace mlir::tessera_rocm
