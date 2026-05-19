//===- tessera-translate.cpp - MLIR translation entry for Tessera --------===//
//
// `tessera-translate-mlir` is the C++ MLIR-side counterpart to the Python
// `tessera-translate` console script.  It is a thin wrapper around
// `mlirTranslateMain` that registers the Tessera dialects on top of the
// MLIR LLVM-IR and SPIR-V translations, so callers can do things like::
//
//   tessera-translate-mlir --mlir-to-llvmir   input.mlir > output.ll
//   tessera-translate-mlir --import-llvm      input.ll   > output.mlir
//   tessera-translate-mlir --serialize-spirv  input.mlir > output.spv
//   tessera-translate-mlir --deserialize-spirv input.spv > output.mlir
//
// The Python `tessera-translate` console script (handled by
// `tessera.cli.translate`) covers the format-export surface (StableHLO
// text, GGUF binary, SafeTensors).  The two tools share the entry name
// stem; the C++ tool installs under `tessera-translate-mlir` to avoid
// the collision per `tools/tessera-translate/README.md`.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"

// Tessera dialects (built-time conditional — non-Tessera builds still
// produce a working mlir-translate stand-in).
#ifdef TESSERA_HAVE_CORE_TESSERA_IR
#include "Tessera/IR/Dialects.h"
#endif

#ifdef TESSERA_HAVE_NEIGHBORS
#include "tessera/Dialect/Neighbors/IR/NeighborsDialect.h"
#endif

#ifdef TESSERA_HAVE_TPP
#include "tpp/InitTPP.h"
#endif

#ifdef TESSERA_HAVE_APPLE_BACKEND
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"
#endif

// MLIR side.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

int main(int argc, char **argv) {
  // Register the standard --mlir-to-llvmir / --import-llvm translations
  // plus --serialize-spirv / --deserialize-spirv.
  mlir::registerToLLVMIRTranslation();
  mlir::registerFromLLVMIRTranslation();
  mlir::registerToSPIRVTranslation();
  mlir::registerFromSPIRVTranslation();

  mlir::DialectRegistry registry;
  // Register only the dialects we actually link.  This is a deliberate
  // narrower scope than `mlir::registerAllDialects()`: the translation
  // surface is well-defined by the upstream LLVM-IR / SPIR-V import/export
  // plus the Tessera dialects below.
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect,
                  mlir::ROCDL::ROCDLDialect,
                  mlir::spirv::SPIRVDialect>();
  // Hook the LLVM-IR translation patterns for the dialects above.
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);

  // Tessera-side dialects.
#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  tessera::registerTesseraDialects(registry);
#endif
#ifdef TESSERA_HAVE_NEIGHBORS
  tessera::neighbors::registerNeighborsDialect(registry);
#endif
#ifdef TESSERA_HAVE_TPP
  tessera::tpp::registerTPPDialect(registry);
#endif
#ifdef TESSERA_HAVE_APPLE_BACKEND
  tessera::apple::registerTesseraAppleBackendDialects(registry);
#endif

  return failed(mlir::mlirTranslateMain(argc, argv,
                                        "Tessera MLIR translation tool"));
}
