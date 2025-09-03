#pragma once
#include <string>
#include "mlir/IR/Types.h"

namespace mlir::tessera_rocm {

// Simple chooser for common MFMA intrinsics keyed by element types + accum
std::string chooseMFMAIntrinsic(mlir::Type aTy, mlir::Type bTy, mlir::Type accTy, llvm::StringRef mcpu);

} // namespace mlir::tessera_rocm
