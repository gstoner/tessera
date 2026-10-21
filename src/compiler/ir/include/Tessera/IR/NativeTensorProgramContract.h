// Shared typed host program ownership and launch verification.
// Used at both Tile and NVIDIA Target boundaries; no numerical family dispatch.
#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
  if (auto view = value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) return view.getSource();
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
struct ShapeMemoryContract {
  llvm::SmallVector<mlir::Value> owned, dimensions;
  llvm::SmallVector<mlir::MemRefType> capacities;
  llvm::SmallVector<llvm::SmallVector<int64_t>> bindings;

  int64_t index(mlir::Value value) const {
    if (auto view = value.getDefiningOp<mlir::memref::ReinterpretCastOp>())
      value = view.getSource();
    auto found = llvm::find(owned, value);
    return found == owned.end() ? -1 : std::distance(owned.begin(), found);
  }
  mlir::MemRefType capacity(mlir::Value value) const {
    int64_t id = index(value);
    return id < 0 ? mlir::MemRefType{} : capacities[id];
  }
  bool matches(mlir::Value value, mlir::Type raw) const {
    using namespace mlir;
    int64_t id = index(value);
    auto type = dyn_cast<ShapedType>(raw);
    if (id < 0 || !type || !type.hasRank() || type.getRank() != 2 ||
        type.getElementType() != capacities[id].getElementType()) return false;
    for (unsigned axis = 0; axis < 2; ++axis)
      if (type.getDimSize(axis) != (bindings[id][axis] < 0 ?
          capacities[id].getDimSize(axis) : ShapedType::kDynamic)) return false;
    return true;
  }
  static bool integer(mlir::Value value, int64_t expected) {
    auto op = value.getDefiningOp<mlir::arith::ConstantOp>();
    auto attr = op ? mlir::dyn_cast<mlir::IntegerAttr>(op.getValue()) : mlir::IntegerAttr{};
    return attr && attr.getValue().getBitWidth() <= 64 && attr.getInt() == expected;
  }
  static bool folded(mlir::OpFoldResult value, mlir::Value dynamic, int64_t fixed) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(value)) {
      auto number = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      return !dynamic && number && number.getInt() == fixed;
    }
    auto actual = mlir::cast<mlir::Value>(value);
    return dynamic ? actual == dynamic : integer(actual, fixed);
  }
  mlir::LogicalResult verify(mlir::ModuleOp module, mlir::func::FuncOp host,
                             mlir::Operation *anchor) {
    using namespace mlir;
    auto error = [&]() { return anchor->emitOpError("shape frame requires checked capacity, dimension bindings and compact active-view SSA"); };
    auto types = module->getAttrOfType<ArrayAttr>("tessera.native.tensor_program_capacity_types");
    auto axes = module->getAttrOfType<ArrayAttr>("tessera.native.tensor_program_shape_bindings");
    auto variables = module->getAttrOfType<DenseI64ArrayAttr>("tessera.native.tensor_program_shape_variables");
    if (!types || !axes || !variables || variables.empty() || variables.size() % 3 ||
        types.size() != axes.size() || types.size() > 128) return error();
    owned.append(host.getArguments().begin(), host.getArguments().end());
    for (auto alloc : host.getBody().front().getOps<memref::AllocOp>()) owned.push_back(alloc);
    if (owned.size() != types.size()) return error();
    unsigned symbols = variables.size() / 3;
    dimensions.resize(symbols);
    for (auto [i, raw] : llvm::enumerate(types)) {
      auto attr = dyn_cast<TypeAttr>(raw);
      auto cap = attr ? dyn_cast<MemRefType>(attr.getValue()) : MemRefType{};
      auto binding = dyn_cast<DenseI64ArrayAttr>(axes[i]);
      if (!cap || !isPhysicalBuffer(cap) || !binding || binding.size() != 2) return error();
      capacities.push_back(cap);
      bindings.emplace_back(binding.asArrayRef().begin(), binding.asArrayRef().end());
      for (unsigned axis = 0; axis < 2; ++axis) {
        int64_t symbol = binding[axis];
        if (symbol < -1 || symbol >= symbols ||
            (symbol >= 0 && variables[symbol * 3 + 2] != cap.getDimSize(axis))) return error();
      }
      if (i < host.getNumArguments()) {
        if (!matches(owned[i], owned[i].getType())) return error();
        auto active = dyn_cast<MemRefType>(owned[i].getType());
        if (!active || !active.getLayout().isIdentity() || active.getMemorySpace()) return error();
      } else if (owned[i].getType() != cap) return error();
    }
    auto guarded = [&](arith::CmpIPredicate predicate, Value lhs, Value rhs,
                       Operation *before) {
      for (auto assertion : host.getBody().front().getOps<cf::AssertOp>()) {
        if (!assertion->isBeforeInBlock(before)) continue;
        auto cmp = assertion.getArg().getDefiningOp<arith::CmpIOp>();
        if (cmp && cmp.getPredicate() == predicate && cmp.getLhs() == lhs && cmp.getRhs() == rhs) return true;
      }
      return false;
    };
    for (unsigned symbol = 0; symbol < symbols; ++symbol) {
      int64_t arg = variables[symbol * 3], axis = variables[symbol * 3 + 1];
      int64_t bound = variables[symbol * 3 + 2];
      if (arg < 0 || arg >= host.getNumArguments() || axis < 0 || axis > 1 ||
          bound <= 0 || bound > INT32_MAX || bindings[arg][axis] != symbol) return error();
      for (auto dim : host.getBody().front().getOps<memref::DimOp>())
        if (dim.getSource() == host.getArgument(arg) && dim.getConstantIndex() == axis) {
          if (dimensions[symbol]) return error();
          dimensions[symbol] = dim;
        }
      Value extent = dimensions[symbol];
      if (!extent) return error();
      bool checked = false;
      for (auto assertion : host.getBody().front().getOps<cf::AssertOp>()) {
        if (!assertion->isBeforeInBlock(anchor)) continue;
        auto both = assertion.getArg().getDefiningOp<arith::AndIOp>();
        if (!both) continue;
        auto positive = both.getLhs().getDefiningOp<arith::CmpIOp>();
        auto fits = both.getRhs().getDefiningOp<arith::CmpIOp>();
        checked |= positive && fits && positive.getPredicate() == arith::CmpIPredicate::ugt &&
          positive.getLhs() == extent && integer(positive.getRhs(), 0) &&
          fits.getPredicate() == arith::CmpIPredicate::ule && fits.getLhs() == extent && integer(fits.getRhs(), bound);
      }
      if (!checked) return error();
    }
    for (auto dim : host.getBody().front().getOps<memref::DimOp>()) {
      auto arg = dyn_cast<BlockArgument>(dim.getSource());
      auto axis = dim.getConstantIndex();
      if (!arg || arg.getOwner() != &host.getBody().front() || !axis || *axis < 0 || *axis > 1) return error();
      int64_t symbol = bindings[arg.getArgNumber()][*axis];
      if (symbol < 0 || (dim.getResult() != dimensions[symbol] &&
          !guarded(arith::CmpIPredicate::eq, dim, dimensions[symbol], anchor))) return error();
    }
    for (auto view : host.getBody().front().getOps<memref::ReinterpretCastOp>()) {
      int64_t id = index(view.getSource());
      if (id < host.getNumArguments() || id < 0 || !view.getSource().getDefiningOp<memref::AllocOp>() ||
          !matches(view, view.getType()) || !view.getType().getLayout().isIdentity() ||
          view.getType().getMemorySpace()) return error();
      auto sizes = view.getMixedSizes(), strides = view.getMixedStrides(), offsets = view.getMixedOffsets();
      if (sizes.size() != 2 || strides.size() != 2 || offsets.size() != 1 ||
          !folded(offsets[0], {}, 0) || !folded(strides[1], {}, 1)) return error();
      for (unsigned axis = 0; axis < 2; ++axis) {
        int64_t symbol = bindings[id][axis];
        if (!folded(sizes[axis], symbol < 0 ? Value{} : dimensions[symbol], capacities[id].getDimSize(axis))) return error();
      }
      int64_t col = bindings[id][1];
      if (!folded(strides[0], col < 0 ? Value{} : dimensions[col], capacities[id].getDimSize(1))) return error();
      bool volumeGuard = false;
      for (auto assertion : host.getBody().front().getOps<cf::AssertOp>()) {
        if (!assertion->isBeforeInBlock(view)) continue;
        auto cmp = assertion.getArg().getDefiningOp<arith::CmpIOp>();
        auto product = cmp ? cmp.getLhs().getDefiningOp<arith::MulIOp>() : arith::MulIOp{};
        if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ule || !product ||
            !integer(cmp.getRhs(), capacities[id].getNumElements())) continue;
        auto factors = product->getOperands();
        bool same = true;
        for (unsigned axis = 0; axis < 2; ++axis) {
          int64_t symbol = bindings[id][axis];
          same &= symbol < 0 ? integer(factors[axis], capacities[id].getDimSize(axis)) : factors[axis] == dimensions[symbol];
        }
        volumeGuard |= same;
      }
      if (!volumeGuard) return error();
    }
    return success();
  }
};

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
      (moduleSchema.getInt() != 1 && moduleSchema.getInt() != 2 && moduleSchema.getInt() != 3))
    return emitOpError("requires native program module schema 1, 2 or 3");
  bool bounded = moduleSchema.getInt() == 2;
  bool shaped = moduleSchema.getInt() == 3;
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
    if (isa<memref::AllocOp,memref::DimOp,memref::SubViewOp,memref::ReinterpretCastOp,arith::ConstantOp,arith::CmpIOp,arith::AndIOp,arith::MulIOp,cf::AssertOp>(candidate) &&
        failed(mlir::verify(&candidate))) return failure();
  ShapeMemoryContract shape;
  if (shaped && failed(shape.verify(module, host, op))) return failure();
  if (shaped) {
    auto bindings = member->getAttrOfType<DenseI64ArrayAttr>(
        "tessera.native.tensor_program_dimension_bindings");
    auto capacities = member->getAttrOfType<ArrayAttr>(
        "tessera.native.tensor_program_capacity_types");
    if (!bindings || bindings.size() != 2 * buffers.size() ||
        !capacities || capacities.size() != buffers.size())
      return emitOpError("shape member requires complete semantic dimension and capacity bindings");
    for (auto [i, buffer] : llvm::enumerate(buffers)) {
      int64_t id = shape.index(buffer);
      auto type = dyn_cast<TypeAttr>(capacities[i]);
      if (id < 0 || !type || type.getValue() != shape.capacity(buffer))
        return emitOpError("shape member capacity binding differs from its original buffer");
      for (unsigned axis = 0; axis < 2; ++axis)
        if (bindings[i * 2 + axis] != shape.bindings[id][axis])
          return emitOpError("shape member semantic dimension binding differs from its original buffer");
    }
  }
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
  for (Value argument : host.getArguments())
    if (!(shaped ? shape.capacity(argument) : physical(argument.getType())))
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
  for (Value argument : host.getArguments()) storage.push_back(shaped ? shape.capacity(argument) : physical(argument.getType()));
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
    if (shaped) return shape.matches(buffer, active);
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
    if (!(shaped ? shape.matches(buffer, buffer.getType()) : bool(physical(buffer.getType()))) ||
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
      if (shaped) activeMetadata = isa<memref::DimOp,memref::ReinterpretCastOp,arith::ConstantOp,arith::CmpIOp,arith::AndIOp,arith::MulIOp,cf::AssertOp>(operation);
      if (bounded) if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        auto value = dyn_cast<IntegerAttr>(constant.getValue());
        activeMetadata = constant.getType().isIndex() && value && value.getInt() == 0 &&
            constant.getResult().hasOneUse() &&
            isa<memref::DimOp>(*constant.getResult().getUsers().begin());
      }
      if (!activeMetadata) return emitOpError("program contains an unaccounted memory effect");
    }
    if (shaped) if (auto release = dyn_cast<memref::DeallocOp>(operation)) {
      if (!release.getMemref().getDefiningOp<memref::AllocOp>())
        return emitOpError("shape program may retire only its directly owned private allocations");
    }
    if (auto allocation = dyn_cast<memref::AllocOp>(operation)) {
      unsigned writes = 0, releases = 0;
      auto type = allocation.getType();
      if (!isPhysicalBuffer(type))
        return emitOpError("program allocations require static compact storage and positive ABI capacity");
      SmallVector<Value> aliases{allocation.getResult()};
      for (Operation *user : allocation.getResult().getUsers())
        if (auto view = dyn_cast<memref::SubViewOp>(user)) aliases.push_back(view.getResult());
        else if (shaped) if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) aliases.push_back(view.getResult());
      for (Value alias : aliases) for (OpOperand &use : alias.getUses()) {
        if (auto *invocation = use.getOwner(); isInvocation(invocation)) {
          if (use.getOperandNumber() == unsigned(invocation->getAttrOfType<IntegerAttr>("input_count").getInt())) ++writes;
        } else if (auto release = dyn_cast<memref::DeallocOp>(use.getOwner())) {
          ++releases;
          if (alias != allocation.getResult() || capacityBase(result.getOperand(0)) == allocation.getResult())
            return emitOpError("returned allocation must remain live and views cannot own deallocation");
          for (Value view : aliases) for (Operation *user : view.getUsers())
            if ((isInvocation(user) || isa<memref::SubViewOp,memref::ReinterpretCastOp>(user)) && !user->isBeforeInBlock(release))
              return emitOpError("private allocation retires before its final reader");
        } else if (!isa<func::ReturnOp>(use.getOwner()) &&
                   !((bounded || shaped) && alias == allocation.getResult() &&
                     (isa<memref::SubViewOp>(use.getOwner()) ||
                      (shaped && isa<memref::ReinterpretCastOp>(use.getOwner())))))
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
    if (!type || type.getValue() != (shaped ? shape.capacity(buffer) : physical(buffer.getType())))
      return emitOpError("native member physical buffer shape or storage differs");
  }
  for (auto [index, type] : llvm::enumerate(kernel.getArgumentTypes()))
    if (index < buffers.size() ? !isa<LLVM::LLVMPointerType>(type) : !type.isInteger(64))
      return emitOpError("native member requires pointer buffers followed by i64 scalars");
  if (!isLaunchGeometry(geometry.asArrayRef()))
    return emitOpError("launch geometry exceeds the native frame CUDA limits");
  if (!bounded && !shaped) {
    for (StringRef key : {"active_buffer_types", "scalar_row_multipliers", "grid_row_multipliers", "grid_row_divisors", "shape_ir"})
      if (op->hasAttr(key) || kernel->hasAttr((Twine("tessera.native.program_") + key).str()))
        return emitOpError("static frame cannot carry bounded active-view metadata");
  }
  if (shaped) {
    auto shapeIR = kernel->getAttrOfType<StringAttr>("tessera.native.program_shape_ir");
    if (!shapeIR || shapeIR.getValue().empty() || op->getAttr("shape_ir") != shapeIR)
      return emitOpError("shape member must retain the selected native shape computation");
    auto activeTypes = kernel->getAttrOfType<ArrayAttr>("tessera.native.program_active_buffer_types");
    if (!activeTypes || activeTypes.size() != buffers.size())
      return emitOpError("shape member must retain its typed active buffer bindings");
    for (auto [buffer, attr] : llvm::zip(buffers, activeTypes)) {
      auto type = dyn_cast<TypeAttr>(attr);
      if (!type || !shape.matches(buffer, type.getValue()))
        return emitOpError("shape member active dimensions differ from its checked bindings");
    }
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
