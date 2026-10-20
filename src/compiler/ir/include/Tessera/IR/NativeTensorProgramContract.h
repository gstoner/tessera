// Shared typed host program ownership and launch verification.
// Used at both Tile and NVIDIA Target boundaries; no numerical family dispatch.
#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
#include <cstdint>

namespace tessera::native_tensor_program {
inline bool isInvocation(mlir::Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "tile.program_invoke" || name == "tessera_nvidia.program_invoke";
}
// The frame ABI carries only positive compact rank-two f32/f16/bf16
// buffers. Check every root and allocation, even one not used by this stage.
inline bool isPhysicalBuffer(mlir::Type raw) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(raw);
  if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
      !type.getLayout().isIdentity() || type.getMemorySpace())
    return false;
  auto element = type.getElementType();
  if (!element.isF16() && !element.isBF16() && !element.isF32()) return false;
  int64_t bytes = element.isF32() ? 4 : 2;
  for (int64_t extent : type.getShape()) {
    if (extent <= 0 || extent > INT32_MAX || bytes > INT64_MAX / extent) return false;
    bytes *= extent;
  }
  return true;
}
// Frame v1 capacity is shared with the retained runtime and Python transport.
// Arithmetic uses a wider intermediate so individually valid buffers cannot
// overflow the aggregate 256-byte-aligned arena before allocation.
inline mlir::LogicalResult verifyFrameCapacity(
    mlir::Operation *anchor, llvm::ArrayRef<mlir::Type> buffers, size_t stages) {
  using namespace mlir;
  if (buffers.size() < 2 || buffers.size() > 128 || stages < 1 || stages > 64)
    return anchor->emitOpError("native frame supports at most 64 stages and 128 buffers");
  __int128 arena = 0;
  for (Type raw : buffers) {
    if (!isPhysicalBuffer(raw))
      return anchor->emitOpError("native frame capacity requires checked physical buffers");
    auto type = cast<MemRefType>(raw);
    __int128 bytes = static_cast<__int128>(type.getDimSize(0)) *
                     type.getDimSize(1) * (type.getElementType().isF32() ? 4 : 2);
    arena = (arena + bytes + 255) & ~static_cast<__int128>(255);
    if (arena > UINT64_MAX)
      return anchor->emitOpError("native frame aligned arena exceeds its 64-bit capacity");
  }
  return success();
}
// Match the versioned CUDA frame owner's grid and block ABI before export.
inline bool isLaunchGeometry(llvm::ArrayRef<int64_t> geometry) {
  if (geometry.size() != 6) return false;
  int64_t threads = 1;
  for (unsigned axis = 0; axis < 3; ++axis) {
    int64_t gridLimit = axis == 0 ? INT32_MAX : 65535;
    int64_t blockLimit = axis == 2 ? 64 : 1024;
    if (geometry[axis] <= 0 || geometry[axis] > gridLimit ||
        geometry[axis + 3] <= 0 || geometry[axis + 3] > blockLimit)
      return false;
    threads *= geometry[axis + 3];
  }
  return threads <= 1024;
}
inline mlir::LogicalResult verifyInvocation(mlir::Operation *op) {
  using namespace mlir;
  auto emitOpError = [&](const char *message) { return op->emitOpError(message); };
  auto host = op->getParentOfType<func::FuncOp>();
  auto schema = host ? host->getAttrOfType<IntegerAttr>(
      "tessera.native.tensor_program") : IntegerAttr{};
  auto module = op->getParentOfType<ModuleOp>();
  auto target = module ? module->getAttrOfType<StringAttr>("tessera.target") : StringAttr{};
  auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr{};
  if (!target || target.getValue() != "nvidia_sm120" || !arch || arch.getValue() != "sm_120")
    return emitOpError("requires exact nvidia_sm120 program target and sm_120 architecture");
  auto moduleSchema = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program");
  if (!moduleSchema || !moduleSchema.getType().isInteger(64) || moduleSchema.getInt() != 1)
    return emitOpError("requires native program module schema 1");
  auto calleeRef = op->getAttrOfType<SymbolRefAttr>("callee");
  if (!calleeRef || calleeRef.getNestedReferences().size() != 1)
    return emitOpError("requires one isolated member::kernel symbol reference");
  auto member = dyn_cast_or_null<ModuleOp>(
      SymbolTable::lookupSymbolIn(module, calleeRef.getRootReference()));
  if (!member)
    return emitOpError("callee is not a visible native program member");
  if (!member->hasAttr("tessera.native.tensor_program_member"))
    return emitOpError("callee must belong to an owned native program member module");
  auto buffers = op->getOperands();
  auto count = op->getAttrOfType<IntegerAttr>("input_count");
  if (!count || !count.getType().isInteger(64))
    return emitOpError("requires an i64 input count");
  int64_t reads = count.getInt();
  if (!schema || !schema.getType().isInteger(64) || schema.getInt() != 1 ||
      !host.getBody().hasOneBlock() ||
      reads < 1 || reads > 2 || buffers.size() != unsigned(reads + 1))
    return emitOpError("requires its typed native program and one fresh output");
  for (Type type : host.getArgumentTypes())
    if (!isPhysicalBuffer(type))
      return emitOpError("program roots require checked static physical buffer types");
  // Lifetime checks inspect other invocations before their own generated
  // verifier may have run. Establish every arity before reading their output
  // or counting allocation writers; malformed later operations must diagnose.
  for (Operation &candidate : host.getBody().front()) {
    if (!isInvocation(&candidate)) continue;
    auto inputs = candidate.getAttrOfType<IntegerAttr>("input_count");
    if (!inputs || !inputs.getType().isInteger(64) || inputs.getInt() < 1 ||
        inputs.getInt() > 2 || candidate.getNumOperands() != unsigned(inputs.getInt() + 1))
      return candidate.emitOpError("requires checked input/output arity before program lifetime verification");
  }
  SmallVector<Type> storage(host.getArgumentTypes());
  size_t stages = 0;
  for (Operation &candidate : host.getBody().front()) {
    if (isInvocation(&candidate)) ++stages;
    if (auto allocation = dyn_cast<memref::AllocOp>(candidate)) {
      if (!isPhysicalBuffer(allocation.getType()))
        return emitOpError("program allocations require static compact storage and positive ABI capacity");
      storage.push_back(allocation.getType());
    }
  }
  if (failed(verifyFrameCapacity(op, storage, stages))) return failure();
  auto output = buffers.back().getDefiningOp<memref::AllocOp>();
  if (!output) return emitOpError("output must be a private or returned allocation");
  llvm::DenseSet<Value> initialized(host.getArguments().begin(), host.getArguments().end());
  llvm::DenseSet<Value> retired;
  for (Operation &operation : host.getBody().front()) {
    if (&operation == op) break;
    if (auto *invoke = &operation; isInvocation(invoke))
      initialized.insert(invoke->getOperands().back());
    if (auto release = dyn_cast<memref::DeallocOp>(operation))
      retired.insert(release.getMemref());
  }
  if (initialized.contains(buffers.back()) || retired.contains(buffers.back()))
    return emitOpError("output must be written exactly once before retirement");
  for (Value input : buffers.take_front(reads))
    if (!initialized.contains(input) || retired.contains(input) || input == buffers.back())
      return emitOpError("input must retain initialized live storage and may not alias the output");
  for (Value buffer : buffers) {
    if (!isPhysicalBuffer(buffer.getType()))
      return emitOpError("requires static compact rank-two buffers with supported storage and positive ABI capacity");
  }
  auto result = dyn_cast<func::ReturnOp>(host.getBody().front().back());
  if (!result || result.getNumOperands() != 1)
    return emitOpError("program must return one owned allocation");
  Operation *lastInvoke = nullptr;
  for (Operation &operation : host.getBody().front()) {
    if (auto *invoke = &operation; isInvocation(invoke)) lastInvoke = invoke;
    else if (!isa<memref::AllocOp,memref::DeallocOp,func::ReturnOp>(operation))
      return emitOpError("program contains an unaccounted memory effect");
    if (auto allocation = dyn_cast<memref::AllocOp>(operation)) {
      unsigned writes = 0, releases = 0;
      auto type = allocation.getType();
      if (!isPhysicalBuffer(type))
        return emitOpError("program allocations require static compact storage and positive ABI capacity");
      for (OpOperand &use : allocation.getResult().getUses()) {
        if (auto *invocation = use.getOwner(); isInvocation(invocation)) {
          if (use.getOperandNumber() == unsigned(invocation->getAttrOfType<IntegerAttr>("input_count").getInt())) ++writes;
        } else if (auto release = dyn_cast<memref::DeallocOp>(use.getOwner())) {
          ++releases;
          if (allocation.getResult() == result.getOperand(0))
            return emitOpError("returned allocation must remain live");
          for (Operation *user : allocation.getResult().getUsers())
            if (isInvocation(user) && !user->isBeforeInBlock(release))
              return emitOpError("private allocation retires before its final reader");
        } else if (!isa<func::ReturnOp>(use.getOwner()))
          return emitOpError("program allocation escapes its checked ownership uses");
      }
      if (writes != 1 ||
          releases != (allocation.getResult() == result.getOperand(0) ? 0u : 1u))
        return emitOpError("program allocations require one writer and exact ownership retirement");
    }
  }
  if (!lastInvoke || lastInvoke->getOperands().empty() ||
      result.getOperand(0) != lastInvoke->getOperands().back())
    return emitOpError("program must return its final native stage allocation");
  Operation *callee = SymbolTable::lookupNearestSymbolFrom(
      op, op->getAttrOfType<SymbolRefAttr>("callee"));
  if (!callee) return emitOpError("callee is not a visible native program member");
  if (auto graph = dyn_cast<func::FuncOp>(callee)) {
    if (op->getName().getStringRef() != "tile.program_invoke")
      return emitOpError("Target invocation requires a lowered native kernel");
    if (graph.getNumArguments() != unsigned(reads) || graph.getNumResults() != 1)
      return emitOpError("Graph member input/output arity differs");
    SmallVector<Type> types(graph.getArgumentTypes());
    types.push_back(graph.getResultTypes()[0]);
    for (auto [buffer, type] : llvm::zip(buffers, types)) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      auto storage = cast<MemRefType>(buffer.getType());
      if (!tensor || tensor.getShape() != storage.getShape() ||
          tensor.getElementType() != storage.getElementType())
        return emitOpError("Graph member does not match physical buffer types");
    }
    return success();
  }
  auto kernel = dyn_cast<LLVM::LLVMFuncOp>(callee);
  auto scalars = op->getAttrOfType<DenseI64ArrayAttr>("scalars");
  auto geometry = op->getAttrOfType<DenseI64ArrayAttr>("geometry");
  if (!kernel || !kernel->hasAttr("nvvm.kernel") || !scalars || !geometry ||
      geometry.size() != 6 || kernel.getNumArguments() != buffers.size() + scalars.size() ||
      kernel->getAttr("tessera.native.program_scalars") != scalars ||
      kernel->getAttr("tessera.native.program_geometry") != geometry)
    return emitOpError("native launch must retain the selected kernel ABI and geometry");
  auto bindingTypes = kernel->getAttrOfType<ArrayAttr>("tessera.native.program_buffer_types");
  if (!bindingTypes || bindingTypes.size() != buffers.size())
    return emitOpError("native member lost its typed physical buffer bindings");
  for (auto [buffer, binding] : llvm::zip(buffers, bindingTypes)) {
    auto type = dyn_cast<TypeAttr>(binding);
    if (!type || type.getValue() != buffer.getType())
      return emitOpError("native member physical buffer shape or storage differs");
  }
  for (auto [index, type] : llvm::enumerate(kernel.getArgumentTypes()))
    if (index < buffers.size() ? !isa<LLVM::LLVMPointerType>(type) : !type.isInteger(64))
      return emitOpError("native member requires pointer buffers followed by i64 scalars");
  if (!isLaunchGeometry(geometry.asArrayRef()))
    return emitOpError("launch geometry exceeds the native frame CUDA limits");
  return success();
}

} // namespace tessera::native_tensor_program
