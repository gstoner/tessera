#pragma once
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

namespace tessera::row_region {
inline std::string digest(mlir::Operation *op) {
  std::string text;
  llvm::raw_string_ostream os(text);
  op->print(os, mlir::OpPrintingFlags().useLocalScope()); os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
inline std::string sourceDigest(mlir::func::FuncOp fn) {
  auto copy = mlir::cast<mlir::func::FuncOp>(fn->clone());
  copy->removeAttr("tessera.native.sm120_tensor_member");
  copy->removeAttr("tessera.native.sm120_row_producer");
  copy.setSymName("__native_row_body"); copy.setPrivate();
  auto value = digest(copy); copy->destroy(); return value;
}
inline mlir::LogicalResult verifyKernel(mlir::Operation *op) {
  using namespace mlir;
  bool targetBody = op->getName().getStringRef() == "tessera_nvidia.row_region_kernel";
  auto name = op->getAttrOfType<StringAttr>("name");
  auto arch = op->getAttrOfType<StringAttr>("arch");
  auto rows = op->getAttrOfType<IntegerAttr>("rows");
  auto columns = op->getAttrOfType<IntegerAttr>("columns");
  auto storage = op->getAttrOfType<TypeAttr>("storage");
  auto threads = op->getAttrOfType<IntegerAttr>("workgroup_size");
  auto schema = op->getAttrOfType<IntegerAttr>("schema");
  auto bodyHash = op->getAttrOfType<StringAttr>("body_hash");
  auto artifact = op->getAttrOfType<StringAttr>("artifact_hash");
  auto source = op->getAttrOfType<StringAttr>("source_hash");
  auto tileBody = op->getAttrOfType<StringAttr>("tile_body_hash");
  auto hex = [](StringAttr value) {
    return value && value.getValue().size() == 64 && llvm::all_of(value.getValue(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
  };
  if (!name || name.getValue().empty() || !arch || arch.getValue() != "sm_120" ||
      !schema || schema.getInt() != 1 || !rows || rows.getInt() <= 0 || rows.getInt() > INT32_MAX ||
      !columns || columns.getInt() <= 0 || columns.getInt() > INT32_MAX ||
      !storage || (!storage.getValue().isF16() && !storage.getValue().isBF16()) ||
      !threads || threads.getInt() != 128 || !hex(bodyHash) || !hex(artifact) || !hex(source) ||
      (targetBody && !hex(tileBody)) ||
      op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
      op->getRegion(0).front().getNumArguments() != 0 || op->getRegion(0).front().getOperations().size() != 1)
    return op->emitOpError("requires a sealed static two-input/two-output SM120 row body");
  auto fn = dyn_cast<LLVM::LLVMFuncOp>(op->getRegion(0).front().front());
  if (!fn || fn.getName() != name.getValue() || fn.getBody().empty() ||
      fn.getNumArguments() != 6 || !isa<LLVM::LLVMVoidType>(fn.getFunctionType().getReturnType()) ||
      !fn.getArgument(4).getType().isInteger(64) || !fn.getArgument(5).getType().isInteger(64) ||
      fn->hasAttr("nvvm.kernel") || digest(fn) != bodyHash.getValue())
    return op->emitOpError("native row body differs from its sealed six-argument ABI");
  for (unsigned i = 0; i < 4; ++i)
    if (fn.getArgument(i).getType() != LLVM::LLVMPointerType::get(op->getContext()))
      return op->emitOpError("row buffers require explicit generic raw pointer arguments");
  bool valid = true;
  unsigned readRoots = 0, writeRoots = 0;
  llvm::SmallPtrSet<Operation *, 4> views;
  llvm::SmallPtrSet<void *, 4> roots;
  fn.walk([&](Operation *nested) {
    StringRef n = nested->getName().getStringRef();
    if (targetBody && isa<LLVM::LoadOp, LLVM::StoreOp, LLVM::GEPOp>(nested)) {
      Value address;
      Type element;
      bool write = false;
      if (auto load = dyn_cast<LLVM::LoadOp>(nested)) { address = load.getAddr(); element = load.getType(); }
      else if (auto store = dyn_cast<LLVM::StoreOp>(nested)) { address = store.getAddr(); element = store.getValue().getType(); write = true; }
      else { auto gep = cast<LLVM::GEPOp>(nested); address = gep.getBase(); element = gep.getElemType(); }
      while (auto gep = address.getDefiningOp<LLVM::GEPOp>()) address = gep.getBase();
      auto arg = dyn_cast<BlockArgument>(address);
      if (!arg || arg.getOwner() != &fn.getBody().front() || arg.getArgNumber() >= 4 || element != storage.getValue()) valid = false;
      else if (isa<LLVM::LoadOp>(nested)) {
        if (arg.getArgNumber() >= 2) valid = false;
        else readRoots |= 1u << arg.getArgNumber();
      } else if (write) {
        if (arg.getArgNumber() < 2) valid = false;
        else writeRoots |= 1u << arg.getArgNumber();
      }
    } else if (!targetBody && isa<tile::BufferViewOp>(nested)) {
      auto view = cast<tile::BufferViewOp>(nested);
      auto arg = dyn_cast<BlockArgument>(view.getBase());
      auto type = cast<tile::BufferViewType>(view.getResult().getType());
      if (!arg || arg.getOwner() != &fn.getBody().front() || arg.getArgNumber() >= 4 ||
          type.getElementType() != storage.getValue() || type.getOrder() != "row_major" ||
          type.getAccess() != (arg.getArgNumber() < 2 ? "read" : "write")) valid = false;
      else roots.insert(arg.getAsOpaquePointer());
      views.insert(nested);
    } else if ((targetBody || !isa<tile::LoadElementOp, tile::StoreElementOp>(nested)) &&
               n != "llvm.func" && n != "llvm.return" &&
               !n.starts_with("arith.") && !n.starts_with("math.") && !n.starts_with("scf.") &&
               n != "gpu.thread_id" && n != "gpu.block_id" && n != "gpu.block_dim") valid = false;
  });
  if (!valid || (targetBody ? (readRoots != 3 || writeRoots != 12) : (views.size() != 4 || roots.size() != 4)))
    return op->emitOpError("row body must retain four distinct typed views and scalar/reduction SSA only");
  return success();
}
} // namespace tessera::row_region
