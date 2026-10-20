// Shared typed host program ownership and launch verification.
// Used at both Tile and NVIDIA Target boundaries; no numerical family dispatch.
#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
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
// Capacity belongs to an allocation/policy; active borrowed roots and explicit
// subviews keep dynamic memref types. Never promise a full-capacity allocation
// for a smaller resident input merely because its caller declared an M bound.
inline mlir::Value capacityBase(mlir::Value value) {
  if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) return view.getSource();
  return value;
}
inline mlir::MemRefType capacityType(mlir::Type raw, int64_t rowBound) {
  using namespace mlir;
  auto type = dyn_cast<MemRefType>(raw);
  if (!type || type.getRank() != 2 || !type.getLayout().isIdentity() || type.getMemorySpace()) return {};
  SmallVector<int64_t> shape(type.getShape());
  if (ShapedType::isDynamic(shape[0])) {
    if (rowBound <= 0 || ShapedType::isDynamic(shape[1])) return {};
    shape[0] = rowBound;
  }
  auto capacity = MemRefType::get(shape,type.getElementType());
  return isPhysicalBuffer(capacity) ? capacity : MemRefType{};
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
  if (!moduleSchema || !moduleSchema.getType().isInteger(64) ||
      (moduleSchema.getInt() != 1 && moduleSchema.getInt() != 2))
    return emitOpError("requires native program module schema 1 or 2");
  bool bounded = moduleSchema.getInt() == 2;
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
  if (!schema || !schema.getType().isInteger(64) || schema.getInt() != moduleSchema.getInt() ||
      !host.getBody().hasOneBlock() ||
      reads < 1 || reads > 2 || buffers.size() != unsigned(reads + 1))
    return emitOpError("requires its typed native program and one fresh output");
  // A current invocation can precede malformed later view/allocation ops.
  // Validate their standard operand/result invariants before typed accessors
  // or alias/lifetime traversal; never depend on verifier walk order.
  for (Operation &candidate : host.getBody().front())
    if (isa<memref::AllocOp,memref::DimOp,memref::SubViewOp,arith::ConstantOp>(candidate) &&
        failed(mlir::verify(&candidate))) return failure();
  auto rowBound = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound");
  auto rowRoot = module->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_root");
  auto rowBuffers = module->getAttrOfType<DenseI64ArrayAttr>("tessera.native.tensor_program_row_buffers");
  if (bounded && (!rowBound || !rowBound.getType().isInteger(64) ||
      rowBound.getInt() <= 0 || rowBound.getInt() > INT32_MAX))
    return emitOpError("bounded frame requires a positive row capacity");
  auto physical = [&](Type type) -> MemRefType {
    return bounded ? capacityType(type,rowBound.getInt()) :
                     (isPhysicalBuffer(type) ? cast<MemRefType>(type) : MemRefType{});
  };
  for (Type type : host.getArgumentTypes())
    if (!physical(type))
      return emitOpError("program roots require checked static physical buffer types or bounded active views");
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
  SmallVector<Type> storage;
  for (Type type : host.getArgumentTypes()) storage.push_back(physical(type));
  SmallVector<Value> owned(host.getArguments());
  size_t stages = 0;
  for (Operation &candidate : host.getBody().front()) {
    if (isInvocation(&candidate)) ++stages;
    if (auto allocation = dyn_cast<memref::AllocOp>(candidate)) {
      if (!isPhysicalBuffer(allocation.getType()))
        return emitOpError("program allocations require static compact storage and positive ABI capacity");
      storage.push_back(allocation.getType());
      owned.push_back(allocation.getResult());
    }
  }
  if (failed(verifyFrameCapacity(op, storage, stages))) return failure();
  if (bounded) {
    auto i64 = [](IntegerAttr attr) { return attr && attr.getType().isInteger(64); };
    if (!i64(rowBound) || rowBound.getInt() <= 0 || rowBound.getInt() > INT32_MAX ||
        !i64(rowRoot) || rowRoot.getInt() < 0 || rowRoot.getInt() >= host.getNumArguments() ||
        !rowBuffers || static_cast<size_t>(rowBuffers.size()) != storage.size() || rowBuffers[rowRoot.getInt()] != 1)
      return emitOpError("bounded frame requires a checked root row binding and capacity for every buffer");
    for (auto [index, type] : llvm::enumerate(storage)) {
      int64_t flag = rowBuffers[index];
      if ((flag != 0 && flag != 1) ||
          (index < host.getNumArguments() &&
           bool(flag) != cast<MemRefType>(host.getArgument(index).getType()).isDynamicDim(0)) ||
          (flag && cast<MemRefType>(type).getDimSize(0) != rowBound.getInt()))
        return emitOpError("bounded active rows must match their declared buffer capacity");
    }
  } else if (rowBound || rowRoot || rowBuffers)
    return emitOpError("static frame cannot carry bounded row bindings");
  auto activeRows = [&](Value buffer) {
    if (!bounded) return false;
    auto it = llvm::find(owned, capacityBase(buffer));
    return it != owned.end() && rowBuffers[std::distance(owned.begin(),it)] == 1;
  };
  auto matchesActiveType = [&](Value buffer, Type active) {
    auto capacity = physical(buffer.getType());
    auto shape = dyn_cast<ShapedType>(active);
    if (!shape || !shape.hasRank() || shape.getRank() != 2 ||
        shape.getElementType() != capacity.getElementType()) return false;
    if (shape.getDimSize(1) != capacity.getDimSize(1)) return false;
    return activeRows(buffer) ? ShapedType::isDynamic(shape.getDimSize(0)) :
                              shape.getDimSize(0) == capacity.getDimSize(0);
  };
  Value actualRows;
  if (bounded) {
    for (auto dim : host.getBody().front().getOps<memref::DimOp>()) {
      if (actualRows || dim.getSource() != host.getArgument(rowRoot.getInt()) ||
          dim.getConstantIndex() != 0)
        return emitOpError("bounded active views require one original root row dimension");
      actualRows = dim.getResult();
    }
    if (!actualRows)
      return emitOpError("bounded active views require an explicit root row dimension SSA value");
    for (auto view : host.getBody().front().getOps<memref::SubViewOp>()) {
      auto allocation = view.getSource().getDefiningOp<memref::AllocOp>();
      if (!allocation || view.getSourceType().getRank() != 2 || view.getType().getRank() != 2)
        return emitOpError("bounded subview must retain a rank-two owned allocation");
      int64_t offsets[] = {0,0}, sizes[] = {ShapedType::kDynamic,view.getSourceType().getDimSize(1)}, strides[] = {1,1};
      if (!allocation || !activeRows(allocation) ||
          view.getStaticOffsets() != ArrayRef<int64_t>(offsets) ||
          view.getStaticSizes() != ArrayRef<int64_t>(sizes) ||
          view.getStaticStrides() != ArrayRef<int64_t>(strides) ||
          !view.getOffsets().empty() || !view.getStrides().empty() ||
          view.getSizes().size() != 1 || view.getSizes().front() != actualRows ||
          !view.getType().isDynamicDim(0) || !physical(view.getType()) ||
          physical(view.getType()) != allocation.getType())
        return emitOpError("bounded subview must expose the checked active rows of its owned capacity");
    }
  }
  auto output = capacityBase(buffers.back()).getDefiningOp<memref::AllocOp>();
  if (!output) return emitOpError("output must be a private or returned allocation");
  llvm::DenseSet<Value> initialized(host.getArguments().begin(), host.getArguments().end());
  llvm::DenseSet<Value> retired;
  for (Operation &operation : host.getBody().front()) {
    if (&operation == op) break;
    if (auto *invoke = &operation; isInvocation(invoke))
      initialized.insert(capacityBase(invoke->getOperands().back()));
    if (auto release = dyn_cast<memref::DeallocOp>(operation))
      retired.insert(release.getMemref());
  }
  if (initialized.contains(capacityBase(buffers.back())) || retired.contains(capacityBase(buffers.back())))
    return emitOpError("output must be written exactly once before retirement");
  for (Value input : buffers.take_front(reads))
    if (!initialized.contains(capacityBase(input)) || retired.contains(capacityBase(input)) ||
        capacityBase(input) == capacityBase(buffers.back()))
      return emitOpError("input must retain initialized live storage and may not alias the output");
  for (Value buffer : buffers) {
    if (!physical(buffer.getType()) ||
        (bounded && activeRows(buffer) != cast<MemRefType>(buffer.getType()).isDynamicDim(0)))
      return emitOpError("requires compact active buffers with checked owned capacity");
  }
  auto result = dyn_cast<func::ReturnOp>(host.getBody().front().back());
  if (!result || result.getNumOperands() != 1)
    return emitOpError("program must return one owned allocation");
  Operation *lastInvoke = nullptr;
  for (Operation &operation : host.getBody().front()) {
    if (auto *invoke = &operation; isInvocation(invoke)) lastInvoke = invoke;
    else if (!isa<memref::AllocOp,memref::DeallocOp,func::ReturnOp>(operation)) {
      bool activeMetadata = bounded && isa<memref::DimOp,memref::SubViewOp>(operation);
      if (bounded) if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        auto value = dyn_cast<IntegerAttr>(constant.getValue());
        activeMetadata = constant.getType().isIndex() && value && value.getInt() == 0 &&
            constant.getResult().hasOneUse() &&
            isa<memref::DimOp>(*constant.getResult().getUsers().begin());
      }
      if (!activeMetadata) return emitOpError("program contains an unaccounted memory effect");
    }
    if (auto allocation = dyn_cast<memref::AllocOp>(operation)) {
      unsigned writes = 0, releases = 0;
      auto type = allocation.getType();
      if (!isPhysicalBuffer(type))
        return emitOpError("program allocations require static compact storage and positive ABI capacity");
      SmallVector<Value> aliases{allocation.getResult()};
      for (Operation *user : allocation.getResult().getUsers())
        if (auto view = dyn_cast<memref::SubViewOp>(user)) aliases.push_back(view.getResult());
      for (Value alias : aliases) for (OpOperand &use : alias.getUses()) {
        if (auto *invocation = use.getOwner(); isInvocation(invocation)) {
          if (use.getOperandNumber() == unsigned(invocation->getAttrOfType<IntegerAttr>("input_count").getInt())) ++writes;
        } else if (auto release = dyn_cast<memref::DeallocOp>(use.getOwner())) {
          ++releases;
          if (alias != allocation.getResult() || capacityBase(result.getOperand(0)) == allocation.getResult())
            return emitOpError("returned allocation must remain live and views cannot own deallocation");
          for (Value view : aliases) for (Operation *user : view.getUsers())
            if ((isInvocation(user) || isa<memref::SubViewOp>(user)) && !user->isBeforeInBlock(release))
              return emitOpError("private allocation retires before its final reader");
        } else if (!isa<func::ReturnOp>(use.getOwner()) &&
                   !(bounded && alias == allocation.getResult() && isa<memref::SubViewOp>(use.getOwner())))
          return emitOpError("program allocation escapes its checked ownership uses");
      }
      if (writes != 1 ||
          releases != (capacityBase(result.getOperand(0)) == allocation.getResult() ? 0u : 1u))
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
      if (!tensor || tensor.getEncoding() || !matchesActiveType(buffer,tensor))
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
    if (!type || type.getValue() != physical(buffer.getType()))
      return emitOpError("native member physical buffer shape or storage differs");
  }
  for (auto [index, type] : llvm::enumerate(kernel.getArgumentTypes()))
    if (index < buffers.size() ? !isa<LLVM::LLVMPointerType>(type) : !type.isInteger(64))
      return emitOpError("native member requires pointer buffers followed by i64 scalars");
  if (!isLaunchGeometry(geometry.asArrayRef()))
    return emitOpError("launch geometry exceeds the native frame CUDA limits");
  if (!bounded) {
    for (StringRef key : {"active_buffer_types", "scalar_row_multipliers", "grid_row_multipliers", "grid_row_divisors"})
      if (op->hasAttr(key) || kernel->hasAttr((Twine("tessera.native.program_") + key).str()))
        return emitOpError("static frame cannot carry bounded active-view metadata");
  }
  if (bounded) {
    auto activeTypes = kernel->getAttrOfType<ArrayAttr>("tessera.native.program_active_buffer_types");
    auto memberBound = member->getAttrOfType<IntegerAttr>("tessera.native.tensor_program_row_bound");
    if (!activeTypes || activeTypes.size() != buffers.size() || memberBound != rowBound)
      return emitOpError("bounded native member must retain its active types and row capacity");
    for (auto [buffer, attr] : llvm::zip(buffers,activeTypes)) {
      auto type = dyn_cast<TypeAttr>(attr);
      auto view = type ? dyn_cast<MemRefType>(type.getValue()) : MemRefType{};
      if (!view || !view.getLayout().isIdentity() || view.getMemorySpace() ||
          !matchesActiveType(buffer,view))
        return emitOpError("bounded member active views differ from owned capacity bindings");
    }
    for (StringRef key : {"scalar_row_multipliers", "grid_row_multipliers", "grid_row_divisors"}) {
      auto values = op->getAttrOfType<DenseI64ArrayAttr>(key);
      auto selected = kernel->getAttr((Twine("tessera.native.program_") + key).str());
      if (!values || selected != values || values.size() !=
          (key == "scalar_row_multipliers" ? scalars.size() : 3))
        return emitOpError("bounded launch expressions must retain the selected native kernel contract");
    }
    auto scalarRows = op->getAttrOfType<DenseI64ArrayAttr>("scalar_row_multipliers");
    auto gridRows = op->getAttrOfType<DenseI64ArrayAttr>("grid_row_multipliers");
    auto divisors = op->getAttrOfType<DenseI64ArrayAttr>("grid_row_divisors");
    for (auto [index,multiple] : llvm::enumerate(scalarRows.asArrayRef())) {
      __int128 value = static_cast<__int128>(rowBound.getInt()) * multiple;
      if (multiple < 0 || value > INT64_MAX || (multiple && value != scalars[index]))
        return emitOpError("bounded scalar expression differs from its checked capacity value");
    }
    for (unsigned axis = 0; axis < 3; ++axis) {
      __int128 numerator = static_cast<__int128>(rowBound.getInt()) * gridRows[axis];
      if (gridRows[axis] < 0 || divisors[axis] <= 0 || numerator > INT64_MAX ||
          (!gridRows[axis] && divisors[axis] != 1) ||
          (gridRows[axis] && (numerator + divisors[axis] - 1) / divisors[axis] != geometry[axis]))
        return emitOpError("bounded grid expression differs from its checked capacity geometry");
    }
  }
  return success();
}

} // namespace tessera::native_tensor_program
