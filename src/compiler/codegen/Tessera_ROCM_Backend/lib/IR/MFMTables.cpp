#include "TesseraROCM/MFMTables.h"
#include "mlir/IR/BuiltinTypes.h"
using namespace mlir;
using namespace mlir::tessera_rocm;

std::string mlir::tessera_rocm::chooseMFMAIntrinsic(Type aTy, Type bTy, Type accTy, llvm::StringRef mcpu){
  auto fty = dyn_cast<FloatType>(accTy);
  if (!fty || fty.getWidth()!=32) return "llvm.amdgcn.mfma.f32.16x16x16.bf16"; // default

  int aBits = 0;
  if (auto fa = dyn_cast<FloatType>(aTy)) aBits = fa.getWidth();
  // crude per-mcpu tweak
  if (mcpu.startswith("gfx120")) {
    if (aBits==16) return "llvm.amdgcn.mfma.f32.32x32x16.f16";
    if (aBits==32) return "llvm.amdgcn.mfma.f32.32x32x4.f32";
    return "llvm.amdgcn.mfma.f32.32x32x16.bf16";
  }
  if (mcpu.startswith("gfx94")) {
    if (aBits==16) return "llvm.amdgcn.mfma.f32.32x32x8.f16";
    if (aBits==32) return "llvm.amdgcn.mfma.f32.32x32x2.f32";
    return "llvm.amdgcn.mfma.f32.16x16x16.bf16";
  }
  // gfx90a default
  if (aBits==16) return "llvm.amdgcn.mfma.f32.16x16x16.f16";
  if (aBits==32) return "llvm.amdgcn.mfma.f32.32x32x2.f32";
  return "llvm.amdgcn.mfma.f32.16x16x16.bf16";
}
