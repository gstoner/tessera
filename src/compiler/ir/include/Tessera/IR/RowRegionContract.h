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
  copy->removeAttr("tessera.native.rocm_row_producer");
  copy.setSymName("__native_row_body"); copy.setPrivate();
  auto value = digest(copy); copy->destroy(); return value;
}
// Schema 2 is the gfx1201 one-parent gate/up view profile. Schema 1 retains
// its original SM120 four-pointer row/reduction meaning. Schema 3 requires
// BF16 parent/output storage with explicit FP32 computation.
inline mlir::LogicalResult verifyGateUpKernel(mlir::Operation *op) {
  using namespace mlir;
  StringRef opName = op->getName().getStringRef();
  bool target = opName == "tessera_rocm.row_region_kernel";
  auto arch = op->getAttrOfType<StringAttr>("arch");
  auto schema = op->getAttrOfType<IntegerAttr>("schema");
  auto name = op->getAttrOfType<StringAttr>("name");
  auto rows = op->getAttrOfType<IntegerAttr>("rows");
  auto columns = op->getAttrOfType<IntegerAttr>("columns");
  auto storage = op->getAttrOfType<TypeAttr>("storage");
  auto threads = op->getAttrOfType<IntegerAttr>("workgroup_size");
  auto hex = [](Attribute a) {
    auto s = dyn_cast_or_null<StringAttr>(a);
    return s && s.getValue().size() == 64 && llvm::all_of(s.getValue(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
  };
  if ((!target && opName != "tile.row_region_kernel") ||
      !arch || arch.getValue() != "gfx1201" || !schema || (schema.getInt() != 2 && schema.getInt() != 3) ||
      !name || name.getValue().empty() || !rows || rows.getInt() <= 0 || rows.getInt() > INT32_MAX ||
      !columns || columns.getInt() <= 0 || columns.getInt() > INT32_MAX / 2 ||
      rows.getInt() > INT64_MAX / 8 / columns.getInt() ||
      rows.getInt() * columns.getInt() > (int64_t(1) << 31) / 12 ||
      !storage || (schema.getInt() == 2 ? !storage.getValue().isF32() : !storage.getValue().isBF16()) || !threads || threads.getInt() != 256 ||
      !hex(op->getAttr("body_hash")) || !hex(op->getAttr("source_hash")) ||
      !hex(op->getAttr("artifact_hash")) || (target && !hex(op->getAttr("tile_body_hash"))) ||
      op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
      op->getRegion(0).front().getNumArguments() != 0 ||
      op->getRegion(0).front().getOperations().size() != 1)
    return op->emitOpError("requires the sealed gfx1201 schema-2 FP32 or schema-3 BF16 parent-view profile");
  auto fn = dyn_cast<LLVM::LLVMFuncOp>(op->getRegion(0).front().front());
  auto ptr = LLVM::LLVMPointerType::get(op->getContext());
  if (!fn || fn.getName() != name.getValue() || !fn.getBody().hasOneBlock() ||
      fn.getNumArguments() != 4 || fn.getArgument(0).getType() != ptr ||
      fn.getArgument(1).getType() != ptr || !fn.getArgument(2).getType().isInteger(64) ||
      !fn.getArgument(3).getType().isInteger(64) ||
      !isa<LLVM::LLVMVoidType>(fn.getFunctionType().getReturnType()) ||
      fn->hasAttr("nvvm.kernel") || fn->hasAttr("rocdl.kernel") || fn->hasAttr("passthrough") ||
      digest(fn) != cast<StringAttr>(op->getAttr("body_hash")).getValue())
    return op->emitOpError("parent-view body differs from its sealed two-pointer/gid/count ABI");
  bool valid = true;
  unsigned reads = 0, writes = 0, loads = 0, stores = 0;
  llvm::SmallPtrSet<Operation *, 2> views;
  llvm::SmallPtrSet<void *, 2> roots;
  fn.walk([&](Operation *nested) {
    StringRef n = nested->getName().getStringRef();
    if (target && isa<LLVM::LoadOp,LLVM::StoreOp,LLVM::GEPOp>(nested)) {
      Value address; Type element; bool write = false;
      if (auto load = dyn_cast<LLVM::LoadOp>(nested)) {
        address = load.getAddr(); element = load.getType(); ++loads;
      } else if (auto store = dyn_cast<LLVM::StoreOp>(nested)) {
        address = store.getAddr(); element = store.getValue().getType(); write = true; ++stores;
      } else { auto gep = cast<LLVM::GEPOp>(nested); address = gep.getBase(); element = gep.getElemType(); }
      while (auto gep = address.getDefiningOp<LLVM::GEPOp>()) address = gep.getBase();
      auto arg = dyn_cast<BlockArgument>(address);
      if (!arg || arg.getOwner() != &fn.getBody().front() || arg.getArgNumber() >= 2 || element != storage.getValue()) valid = false;
      else if (isa<LLVM::LoadOp>(nested)) { if (arg.getArgNumber() != 0) valid = false; else reads |= 1; }
      else if (write) { if (arg.getArgNumber() != 1) valid = false; else writes |= 2; }
    } else if (!target && isa<tile::BufferViewOp>(nested)) {
      auto view = cast<tile::BufferViewOp>(nested);
      auto arg = dyn_cast<BlockArgument>(view.getBase());
      auto type = cast<tile::BufferViewType>(view.getResult().getType());
      if (!arg || arg.getOwner() != &fn.getBody().front() || arg.getArgNumber() >= 2 ||
          type.getElementType() != storage.getValue() || type.getOrder() != "row_major" ||
          type.getAccess() != (arg.getArgNumber() == 0 ? "read" : "write")) valid = false;
      else roots.insert(arg.getAsOpaquePointer());
      views.insert(nested);
    } else if (!target && isa<tile::LoadElementOp>(nested)) ++loads;
    else if (!target && isa<tile::StoreElementOp>(nested)) ++stores;
    else if (n != "llvm.func" && n != "llvm.return" && !n.starts_with("arith.") &&
             !n.starts_with("math.") && !n.starts_with("scf.")) valid = false;
  });
  if (!valid || loads != 2 || stores != 1 ||
      (target ? (reads != 1 || writes != 2) : (views.size() != 2 || roots.size() != 2)))
    return op->emitOpError("parent-view body requires two parent reads and one distinct output store");
  return success();
}
// Schema 6 is the SM120 two-input/one-output reverse row profile; schema 1
// retains its original two-input/two-output JVP ABI. Schema 5 is not reused.
// Schema 4 is a distinct mixed-width K16 quantizer, never a gate/up recipe.
inline mlir::LogicalResult verifyBlockQuantKernel(mlir::Operation *op) {
  using namespace mlir;
  bool target=op->getName().getStringRef()=="tessera_rocm.row_region_kernel";
  auto fail=[&]() { return op->emitOpError("requires the sealed gfx1201 schema-4 typed K16 quantizer"); };
  auto arch=op->getAttrOfType<StringAttr>("arch");
  auto name=op->getAttrOfType<StringAttr>("name");
  auto rows=op->getAttrOfType<IntegerAttr>("rows");
  auto columns=op->getAttrOfType<IntegerAttr>("columns");
  auto threads=op->getAttrOfType<IntegerAttr>("workgroup_size");
  auto hex=[](Attribute raw) {
    auto text=dyn_cast_or_null<StringAttr>(raw);
    return text && text.getValue().size()==64 && llvm::all_of(text.getValue(),[](char c) {
      return (c>='0' && c<='9') || (c>='a' && c<='f');
    });
  };
  if ((!target && op->getName().getStringRef()!="tile.row_region_kernel") ||
      !arch || arch.getValue()!="gfx1201" || !name || name.getValue().empty() ||
      op->getAttrOfType<TypeAttr>("storage")!=TypeAttr::get(BFloat16Type::get(op->getContext())) ||
      !rows || rows.getInt()<=0 || rows.getInt()>INT32_MAX ||
      !columns || columns.getInt()<=0 || columns.getInt()>INT32_MAX || columns.getInt()%16 ||
      rows.getInt()>INT32_MAX/(columns.getInt()/16) ||
      !threads || threads.getInt()!=256 ||
      !hex(op->getAttr("source_hash")) || !hex(op->getAttr("artifact_hash")) ||
      !hex(op->getAttr("body_hash")) || (target && !hex(op->getAttr("tile_body_hash"))) || op->getNumRegions()!=1 ||
      !op->getRegion(0).hasOneBlock() || op->getRegion(0).front().getNumArguments() ||
      op->getRegion(0).front().getOperations().size()!=1) return fail();
  auto fn=dyn_cast<LLVM::LLVMFuncOp>(op->getRegion(0).front().front());
  if (!fn || fn.getName()!=name.getValue() || !fn.getBody().hasOneBlock() ||
      fn.getNumArguments()!=5 || !isa<LLVM::LLVMVoidType>(fn.getFunctionType().getReturnType()) ||
      fn->hasAttr("nvvm.kernel") || fn->hasAttr("rocdl.kernel") || fn->hasAttr("passthrough") ||
      digest(fn)!=cast<StringAttr>(op->getAttr("body_hash")).getValue()) return fail();
  for (unsigned i=0;i<5;++i)
    if (i<3 ? !isa<LLVM::LLVMPointerType>(fn.getArgument(i).getType()) :
              !fn.getArgument(i).getType().isInteger(64)) return fail();
  bool valid=true;
  unsigned roots=0,loads=0,stores=0;
  fn.walk([&](Operation *nested) {
    auto n=nested->getName().getStringRef();
    if (target && isa<LLVM::LoadOp,LLVM::StoreOp,LLVM::GEPOp>(nested)) {
      Value address; Type element; bool read=false,write=false;
      if (auto load=dyn_cast<LLVM::LoadOp>(nested)) {
        address=load.getAddr(); element=load.getType(); read=true; ++loads;
      } else if (auto store=dyn_cast<LLVM::StoreOp>(nested)) {
        address=store.getAddr(); element=store.getValue().getType(); write=true; ++stores;
      } else {
        auto gep=cast<LLVM::GEPOp>(nested); address=gep.getBase(); element=gep.getElemType();
      }
      while (auto gep=address.getDefiningOp<LLVM::GEPOp>()) address=gep.getBase();
      auto arg=dyn_cast<BlockArgument>(address);
      if (!arg || arg.getOwner()!=&fn.getBody().front() || arg.getArgNumber()>=3) { valid=false; return; }
      unsigned i=arg.getArgNumber();
      if ((read && i!=0) || (write && i==0) ||
          (i==0 ? !element.isBF16() : i==1 ? !element.isInteger(8) : !element.isF32())) valid=false;
      roots|=1u<<i;
    } else if (!target && isa<tile::BufferViewOp>(nested)) {
      auto view=cast<tile::BufferViewOp>(nested);
      auto arg=dyn_cast<BlockArgument>(view.getBase());
      auto type=cast<tile::BufferViewType>(view.getType());
      if (!arg || arg.getOwner()!=&fn.getBody().front() || arg.getArgNumber()>=3) { valid=false; return; }
      unsigned i=arg.getArgNumber();
      if ((roots & (1u<<i)) || type.getOrder()!="row_major" ||
          type.getAccess()!=(i==0?"read":"write") ||
          (i==0 ? !type.getElementType().isBF16() :
           i==1 ? !isa<Float8E4M3FNType>(type.getElementType()) : !type.getElementType().isF32())) valid=false;
      roots|=1u<<i;
    } else if (!target && isa<tile::LoadElementOp>(nested)) {
      auto load=cast<tile::LoadElementOp>(nested);
      ++loads;
      auto view=load->getOperand(0).getDefiningOp<tile::BufferViewOp>();
      if (!view || view.getBase()!=fn.getArgument(0)) valid=false;
    } else if (!target && isa<tile::StoreElementOp>(nested)) {
      auto store=cast<tile::StoreElementOp>(nested);
      ++stores;
      auto view=store->getOperand(0).getDefiningOp<tile::BufferViewOp>();
      if (!view || (view.getBase()!=fn.getArgument(1) && view.getBase()!=fn.getArgument(2))) valid=false;
    } else if (n!="llvm.func" && n!="llvm.return" && !n.starts_with("arith.") &&
               !n.starts_with("math.") && !n.starts_with("scf.")) valid=false;
  });
  if (!valid || roots!=7 || loads!=2 || stores!=2) return fail();
  return success();
}
inline mlir::LogicalResult verifyKernel(mlir::Operation *op) {
  auto schemaProfile = op->getAttrOfType<mlir::IntegerAttr>("schema");
  if (schemaProfile && schemaProfile.getInt()==4) return verifyBlockQuantKernel(op);
  if (schemaProfile && (schemaProfile.getInt() == 2 || schemaProfile.getInt() == 3))
    return verifyGateUpKernel(op);
  using namespace mlir;
  bool rocmProfile = schemaProfile && schemaProfile.getInt() == 7;
  bool boundedProfile = schemaProfile && schemaProfile.getInt() == 8;
  bool singleOutput = schemaProfile && (schemaProfile.getInt() == 6 || boundedProfile || rocmProfile);
  unsigned buffers = singleOutput ? 3 : 4;
  bool nvidiaTargetBody = op->getName().getStringRef() == "tessera_nvidia.row_region_kernel";
  bool rocmTargetBody = op->getName().getStringRef() == "tessera_rocm.row_region_kernel";
  bool targetBody = nvidiaTargetBody || rocmTargetBody;
  if ((nvidiaTargetBody && rocmProfile) || (rocmTargetBody && !rocmProfile))
    return op->emitOpError("native row Target ownership differs from its schema");
  if (!targetBody && op->getName().getStringRef() != "tile.row_region_kernel")
    return op->emitOpError("schema-1 row profile requires Tile or NVIDIA Target ownership");
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
  if (!name || name.getValue().empty() || !arch ||
      (rocmProfile ? (arch.getValue() != "gfx1151" && arch.getValue() != "gfx1201") : arch.getValue() != "sm_120") ||
      !schema || (schema.getInt() != 1 && schema.getInt() != 6 && schema.getInt() != 7 && schema.getInt() != 8) || !rows || rows.getInt() <= 0 || rows.getInt() > INT32_MAX ||
      !columns || columns.getInt() <= 0 || columns.getInt() > INT32_MAX ||
      !storage || (!storage.getValue().isF16() && !storage.getValue().isBF16()) ||
      !threads || threads.getInt() != 128 || !hex(bodyHash) || !hex(artifact) || !hex(source) ||
      (targetBody && !hex(tileBody)) ||
      op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
      op->getRegion(0).front().getNumArguments() != 0 || op->getRegion(0).front().getOperations().size() != 1)
    return op->emitOpError("requires a sealed native SM120 schema-1/6/8 or ROCm schema-7 row body");
  auto multiple = op->getAttrOfType<IntegerAttr>("row_multiple");
  if (boundedProfile ? (!multiple || !multiple.getType().isInteger(64) ||
          multiple.getInt() <= 0 || multiple.getInt() > rows.getInt() ||
          rows.getInt() % multiple.getInt() != 0)
                     : op->hasAttr("row_multiple"))
    return op->emitOpError("bounded-query rows require their positive capacity-dividing logical prefix");
  auto fn = dyn_cast<LLVM::LLVMFuncOp>(op->getRegion(0).front().front());
  if (!fn || fn.getName() != name.getValue() || fn.getBody().empty() ||
      fn.getNumArguments() != buffers + 2 || !isa<LLVM::LLVMVoidType>(fn.getFunctionType().getReturnType()) ||
      !fn.getArgument(buffers).getType().isInteger(64) || !fn.getArgument(buffers+1).getType().isInteger(64) ||
      fn->hasAttr("nvvm.kernel") || digest(fn) != bodyHash.getValue())
    return op->emitOpError("native row body differs from its sealed profile-specific pointer/dimension ABI");
  for (unsigned i = 0; i < buffers; ++i)
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
      if (!arg || arg.getOwner() != &fn.getBody().front() || arg.getArgNumber() >= buffers || element != storage.getValue()) valid = false;
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
      if (!arg || arg.getOwner() != &fn.getBody().front() || arg.getArgNumber() >= buffers ||
          type.getElementType() != storage.getValue() || type.getOrder() != "row_major" ||
          type.getAccess() != (arg.getArgNumber() < 2 ? "read" : "write")) valid = false;
      else roots.insert(arg.getAsOpaquePointer());
      views.insert(nested);
    } else if ((targetBody || !isa<tile::LoadElementOp, tile::StoreElementOp>(nested)) &&
               n != "llvm.func" && n != "llvm.return" &&
               !n.starts_with("arith.") && !n.starts_with("math.") && !n.starts_with("scf.") &&
               n != "gpu.thread_id" && n != "gpu.block_id" && n != "gpu.block_dim") valid = false;
  });
  if (!valid || (targetBody ? (readRoots != 3 || writeRoots != (singleOutput ? 4u : 12u)) : (views.size() != buffers || roots.size() != buffers)))
    return op->emitOpError("row body must retain profile-specific distinct typed views and scalar/reduction SSA only");
  return success();
}
} // namespace tessera::row_region
