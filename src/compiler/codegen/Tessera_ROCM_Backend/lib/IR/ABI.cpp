#include "TesseraROCM/ABI.h"
#include "mlir/IR/BuiltinAttributes.h"
using namespace mlir;
using namespace mlir::tessera_rocm;

void mlir::tessera_rocm::annotateKernelABI(func::FuncOp fn, const ABIConfig &cfg){
  auto ctx = fn.getContext();
  auto wg = cfg.wgX * cfg.wgY * cfg.wgZ;
  fn->setAttr("amdgpu-flat-work-group-size",
    ArrayAttr::get(ctx, {IntegerAttr::get(IntegerType::get(ctx,32),(int64_t)wg),
                         IntegerAttr::get(IntegerType::get(ctx,32),(int64_t)wg)}));
  fn->setAttr("tessera.rocm.mcpu", StringAttr::get(ctx, cfg.mcpu));
  if (cfg.ldsBytes)
    fn->setAttr("amdgpu-lds-size", IntegerAttr::get(IntegerType::get(ctx,32),(int64_t)cfg.ldsBytes));
}
