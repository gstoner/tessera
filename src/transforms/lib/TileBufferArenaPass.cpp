// TileBufferArenaPass.cpp — realize the buffer-reuse plan into a concrete
// shared-memory arena layout (Workstream H / W3 follow-on, 2026-07-08).
//
// TileBufferReusePass *assigns* disjoint-live-range buffers to reuse groups
// (`tile.buffer_group`). This pass is the first real CONSUMER of that metadata:
// it lays each group out in a per-space arena and stamps a concrete byte offset
// on every allocation —
//
//   tile.alloc_shared %buf {tile.buffer_group = 0, tile.smem_offset = 0}   : ...
//   tile.alloc_shared %buf {tile.buffer_group = 1, tile.smem_offset = 512} : ...
//   tile.alloc_shared %buf {tile.buffer_group = 0, tile.smem_offset = 0}   : ...
//
// so two buffers in the same group land at the SAME offset (the aliasing the
// reuse decision promised is now realized), and the func records the total arena
// bytes. This is exactly the form a shared-memory backend emits directly
// (`__shared__ char arena[N]; T* buf = (T*)(arena + offset)`), so it turns the
// group id from bookkeeping into an actionable allocation plan — the hardware-free
// half of the consumer (Decision #19), ahead of any HIP/PTX emission.
//
// SMEM (`tile.alloc_shared` → `tile.smem_offset` / `tile.smem_arena_bytes`) and
// TMEM (`tile.tmem.alloc` → `tile.tmem_offset` / `tile.tmem_arena_bytes`) are laid
// out in SEPARATE arenas — they are distinct physical spaces (the reuse pass
// already keeps them in distinct groups).

#include "Tessera/Transforms/Passes.h"
#include "TileMemrefLifetime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

using namespace mlir;

namespace {

constexpr StringRef kGroupAttr = "tile.buffer_group";

static bool isSharedAlloc(Operation *op) {
  return op->getName().getStringRef() == "tile.alloc_shared";
}
static bool isTmemAlloc(Operation *op) {
  return op->getName().getStringRef() == "tile.tmem.alloc";
}

static int64_t staticByteSize(Value v) {
  return tessera::memory::staticBytes(v);
}

// Natural alignment (bytes) of a memref's element — a backend casts
// `arena + offset` to `T*`, so each group's offset must be a multiple of this or
// the typed access is misaligned. Scalar alignment = element byte width.
static int64_t elementAlign(Value v) {
  auto mr = dyn_cast<MemRefType>(v.getType());
  if (!mr || !mr.getElementType().isIntOrFloat())
    return 1;
  int64_t bits = mr.getElementType().getIntOrFloatBitWidth();
  return bits > 0 ? (bits + 7) / 8 : 1;
}

static bool hasDynamicShape(Operation *op) {
  auto mr = dyn_cast<MemRefType>(op->getOperand(0).getType());
  return mr && !mr.hasStaticShape();
}

struct TileBufferArena
    : public PassWrapper<TileBufferArena, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBufferArena)

  StringRef getArgument() const override { return "tessera-tile-buffer-arena"; }
  StringRef getDescription() const override {
    return "Realize the tile.buffer_group reuse plan into a concrete per-space "
           "shared-memory arena: stamp tile.smem_offset / tile.tmem_offset on each "
           "alloc (same-group buffers share an offset) + the arena byte size on "
           "the func. The first consumer of TileBufferReusePass's metadata.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, func::FuncDialect,
                    gpu::GPUDialect, cf::ControlFlowDialect, DLTIDialect, nvgpu::NVGPUDialect>();
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *fn) {
      if (isa<func::FuncOp, gpu::GPUFuncOp>(fn) && fn->getNumRegions())
        layoutRegion(fn);
    });
  }

  // Lay out one space's arena: group -> max member size, offset = cumsum in
  // ascending group-id order (deterministic). Stamps `offsetAttr` on each alloc
  // and returns the total arena bytes. Only groups whose every member has a known
  // static size are placed (an unknown-size group is skipped — no false offset).
  int64_t layoutSpace(const SmallVector<Operation *> &allocs,
                      StringRef offsetAttr, OpBuilder &b) {
    // group id -> max static byte size (-1 if any member is unknown) + max
    // element alignment (so a mixed-dtype arena keeps each group typed-aligned).
    llvm::DenseMap<int64_t, int64_t> groupBytes, groupAlign;
    SmallVector<int64_t> order;                 // ascending unique group ids
    for (Operation *op : allocs) {
      int64_t g = op->getAttrOfType<IntegerAttr>(kGroupAttr).getInt();
      int64_t sz = staticByteSize(op->getOperand(0));
      int64_t al = elementAlign(op->getOperand(0));
      auto it = groupBytes.find(g);
      if (it == groupBytes.end()) {
        groupBytes[g] = sz;
        groupAlign[g] = al;
        order.push_back(g);
      } else {
        groupAlign[g] = std::max(groupAlign[g], al);
        if (sz < 0 || it->second < 0)
          it->second = -1;                      // group poisoned by an unknown dim
        else
          it->second = std::max(it->second, sz);
      }
    }
    llvm::sort(order);
    llvm::DenseMap<int64_t, int64_t> offset;
    int64_t cursor = 0;
    for (int64_t g : order) {
      if (groupBytes[g] < 0)
        continue;                               // unplaceable — leave unstamped
      int64_t a = std::max<int64_t>(groupAlign[g], 1);
      __int128 aligned = ((static_cast<__int128>(cursor) + a - 1) / a) * a;
      if (aligned + groupBytes[g] > std::numeric_limits<int64_t>::max()) {
        allocs.front()->emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: arena size exceeds signed index range");
        signalPassFailure();
        return -1;
      }
      offset[g] = static_cast<int64_t>(aligned);
      cursor = static_cast<int64_t>(aligned + groupBytes[g]);
    }
    for (Operation *op : allocs) {
      int64_t g = op->getAttrOfType<IntegerAttr>(kGroupAttr).getInt();
      auto it = offset.find(g);
      if (it != offset.end())
        op->setAttr(offsetAttr, b.getI64IntegerAttr(it->second));
    }
    return cursor;
  }

  // View/cast descriptors preserve the arena address space throughout the
  // alias chain; changing only their source operand creates invalid memref IR.
  void retargetAliases(Value value, Attribute space) {
    for (Operation *user : value.getUsers()) {
      if (tessera::memory::viewSource(user) != value) continue;
      for (Value result : user->getResults()) {
        auto type = dyn_cast<MemRefType>(result.getType());
        if (!type) continue;
        result.setType(MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(), space));
        retargetAliases(result, space);
      }
    }
  }

  void layoutRegion(Operation *fn) {
    tessera::memory::Lifetimes lifetimes(fn);
    SmallVector<Operation *> planned;
    fn->walk([&](Operation *op) {
      if (tessera::memory::isMarker(op) && op->hasAttr(kGroupAttr)) planned.push_back(op);
    });
    for (auto [i, op] : llvm::enumerate(planned)) {
      auto type = dyn_cast<MemRefType>(op->getOperand(0).getType());
      if (isSharedAlloc(op) && type) {
        bool unsafeAlias = !type.getLayout().isIdentity();
        if (Operation *def = op->getOperand(0).getDefiningOp())
          unsafeAlias |= static_cast<bool>(tessera::memory::viewSource(def));
        DominanceInfo dominance(fn);
        for (Operation *user : op->getOperand(0).getUsers())
          if (tessera::memory::viewSource(user) && !dominance.properlyDominates(op, user)) unsafeAlias = true;
        SmallVector<Value> aliases{op->getOperand(0)};
        llvm::SmallPtrSet<Value, 16> visited;
        while (!aliases.empty()) {
          Value alias = aliases.pop_back_val();
          if (!visited.insert(alias).second) continue;
          for (Operation *user : alias.getUsers()) {
            if (tessera::memory::viewSource(user) == alias) {
              for (Value result : user->getResults()) aliases.push_back(result);
            } else {
              auto name = user->getName().getStringRef();
              bool known = tessera::memory::isMarker(user) ||
                           isa<memref::LoadOp, memref::StoreOp, memref::DimOp>(user) ||
                           name == "tile.async_copy" || name == "tile.wait_async" ||
                           isa<nvgpu::DeviceAsyncCopyOp>(user) ||
                           name == "tile.mma" || tessera::memory::borrowedCall(user, alias, true);
              unsafeAlias |= !known;
            }
          }
        }
        if (unsafeAlias) {
          op->emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: arena cannot rebase a pre-existing, escaping or nonidentity alias");
          signalPassFailure();
          return;
        }
      }
      for (Operation *other : ArrayRef<Operation *>(planned).take_front(i)) {
        if (op->getName() == other->getName() &&
            op->getAttr(kGroupAttr) == other->getAttr(kGroupAttr) &&
            !lifetimes.disjoint(op, other)) {
          op->emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: arena reuse group lacks a disjoint lifetime proof");
          signalPassFailure();
          return;
        }
      }
    }
    OpBuilder b(fn->getContext());
    SmallVector<Operation *> smem, tmem;
    fn->walk([&](Operation *op) {
      if (!op->hasAttr(kGroupAttr))
        return;
      if (isSharedAlloc(op))
        smem.push_back(op);
      else if (isTmemAlloc(op))
        tmem.push_back(op);
    });
    if (smem.empty() && tmem.empty())
      return;
    bool dynamicSmem = llvm::any_of(smem, hasDynamicShape);
    if (dynamicSmem && isa<gpu::GPUFuncOp>(fn)) {
      auto kernel = cast<gpu::GPUFuncOp>(fn);
      if (!kernel.isKernel() || llvm::any_of(smem, [&](Operation *op) {
            if (op->getBlock() != &kernel.getBody().front() && !lifetimes.get(op).reusable) return true;
            for (auto *parent = op->getParentOp(); parent != fn; parent = parent->getParentOp()) {
              if (auto branch = dyn_cast<scf::IfOp>(parent)) {
                if (!tessera::memory::workgroupUniform(branch.getCondition())) return true;
              } else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
                if (!tessera::memory::workgroupUniform(loop.getLowerBound()) ||
                    !tessera::memory::workgroupUniform(loop.getUpperBound()) ||
                    !tessera::memory::workgroupUniform(loop.getStep())) return true;
              } else return true;
            }
            return false;
          })) {
        fn->emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: dynamic GPU arena requires uniform structured kernel regions");
        signalPassFailure();
        return;
      }
      bool existing = false;
      kernel.walk([&](gpu::DynamicSharedMemoryOp) { existing = true; });
      if (existing) {
        fn->emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: dynamic GPU arena cannot overlap an existing dynamic shared allocation");
        signalPassFailure();
        return;
      }
      materializeDynamicSharedArena(fn, smem, &kernel.getBody().front().front(), b);
      int64_t tmemBytes = layoutSpace(tmem, "tile.tmem_offset", b);
      if (!tmem.empty() && tmemBytes >= 0)
        fn->setAttr("tile.tmem_arena_bytes", b.getI64IntegerAttr(tmemBytes));
      return;
    }
    if (dynamicSmem) {
      auto func = dyn_cast<func::FuncOp>(fn);
      DominanceInfo dominance(func);
      llvm::DenseMap<Block *, SmallVector<Operation *>> byBlock;
      for (Operation *alloc : smem)
        byBlock[alloc->getBlock()].push_back(alloc);

      int64_t arenaRegions = 0;
      bool unresolved = !func;
      for (auto &entry : byBlock) {
        auto &blockAllocs = entry.second;
        llvm::sort(blockAllocs, [](Operation *lhs, Operation *rhs) {
          return lhs->isBeforeInBlock(rhs);
        });

        // A descriptor introduced after an earlier marker cannot size that
        // marker's arena. Start a new cohort at that descriptor's first marker.
        // This also makes mutually-exclusive branch-local descriptors legal:
        // each branch receives an arena in its own dominance region.
        SmallVector<Operation *> cohort;
        Operation *insertionPoint = nullptr;
        auto flushCohort = [&]() {
          if (cohort.empty())
            return;
          materializeDynamicSharedArena(func, cohort, insertionPoint, b);
          ++arenaRegions;
          cohort.clear();
        };
        for (Operation *alloc : blockAllocs) {
          if (!insertionPoint) {
            insertionPoint = alloc;
            cohort.push_back(alloc);
            continue;
          }
          if (!dominance.dominates(alloc->getOperand(0), insertionPoint)) {
            flushCohort();
            insertionPoint = alloc;
          }
          cohort.push_back(alloc);
        }
        flushCohort();
      }
      if (unresolved)
        fn->setAttr("tile.smem_arena_dynamic_unresolved", b.getUnitAttr());
      else
        fn->removeAttr("tile.smem_arena_dynamic_unresolved");
      if (arenaRegions > 0)
        fn->setAttr("tile.smem_arena_regions",
                    b.getI64IntegerAttr(arenaRegions));
      int64_t tmemBytes = layoutSpace(tmem, "tile.tmem_offset", b);
      if (!tmem.empty())
        fn->setAttr("tile.tmem_arena_bytes",
                    b.getI64IntegerAttr(tmemBytes));
      return;
    }
    int64_t smemBytes = layoutSpace(smem, "tile.smem_offset", b);
    int64_t tmemBytes = layoutSpace(tmem, "tile.tmem_offset", b);
    if (smemBytes < 0 || tmemBytes < 0) return;
    if (!smem.empty() && !dynamicSmem)
      fn->setAttr("tile.smem_arena_bytes", b.getI64IntegerAttr(smemBytes));
    if (!tmem.empty())
      fn->setAttr("tile.tmem_arena_bytes", b.getI64IntegerAttr(tmemBytes));
    if (!smem.empty() && !dynamicSmem && smemBytes > 0)
      materializeSharedArena(fn, smem, smemBytes, b);
  }

  // Export the exact kernel layout expression as a native host function and
  // wire all local launch sites to it. Every intermediate is in [0, INT32_MAX]:
  // adding or multiplying two checked operands cannot overflow a 64-bit index.
  LogicalResult materializeLaunchSizer(gpu::GPUFuncOp kernel, Value bytes) {
    auto module = kernel->getParentOfType<gpu::GPUModuleOp>()->getParentOfType<ModuleOp>();
    if (DataLayout(module).getTypeSizeInBits(IndexType::get(&getContext())) != 64)
      return kernel.emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: dynamic launch sizing requires a 64-bit host index");
    OpBuilder host(module.getContext());
    host.setInsertionPoint(kernel->getParentOp());
    auto gpuModule = kernel->getParentOfType<gpu::GPUModuleOp>();
    std::string name = ("__tessera_shared_bytes_" + gpuModule.getName() + "_" + kernel.getName()).str();
    if (SymbolTable::lookupSymbolIn(module, name))
      return kernel.emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: launch sizing symbol already exists");
    auto type = host.getFunctionType(kernel.getFunctionType().getInputs(), {host.getIndexType()});
    auto sizing = func::FuncOp::create(host, kernel.getLoc(), name, type);
    sizing->setAttr("llvm.emit_c_interface", host.getUnitAttr());
    Block *entry = sizing.addEntryBlock();
    Block *invalid = sizing.addBlock();
    host.setInsertionPointToStart(invalid);
    auto failure = arith::ConstantIndexOp::create(host, kernel.getLoc(), -1);
    func::ReturnOp::create(host, kernel.getLoc(), failure.getResult());
    host.setInsertionPointToStart(entry);
    IRMapping mapping;
    for (auto [source, target] : llvm::zip(kernel.getArguments().take_front(type.getNumInputs()), sizing.getArguments()))
      mapping.map(source, target);
    llvm::SmallPtrSet<Value, 16> checked;
    auto check = [&](Value value) {
      if (!checked.insert(value).second) return;
      auto zero = arith::ConstantIndexOp::create(host, kernel.getLoc(), 0);
      auto limit = arith::ConstantIndexOp::create(host, kernel.getLoc(), INT32_MAX);
      auto nonnegative = arith::CmpIOp::create(host, kernel.getLoc(), arith::CmpIPredicate::sge, value, zero);
      auto bounded = arith::CmpIOp::create(host, kernel.getLoc(), arith::CmpIPredicate::sle, value, limit);
      auto valid = arith::AndIOp::create(host, kernel.getLoc(), nonnegative, bounded);
      Block *next = sizing.addBlock();
      cf::CondBranchOp::create(host, kernel.getLoc(), valid, next, ValueRange{}, invalid, ValueRange{});
      host.setInsertionPointToStart(next);
    };
    std::function<Value(Value)> clone = [&](Value value) -> Value {
      if (!value.getType().isIndex()) return {};
      if (mapping.contains(value)) {
        auto result = mapping.lookup(value);
        check(result);
        return result;
      }
      Operation *def = value.getDefiningOp();
      if (!def) return {};
      if (auto dim = dyn_cast<memref::DimOp>(def)) {
        if (!mapping.contains(dim.getSource()) || !clone(dim.getIndex())) return {};
      } else if (isa<arith::ConstantIndexOp>(def)) {
        // no operands
      } else if (isa<arith::AddIOp, arith::MulIOp, arith::MaxUIOp, arith::DivUIOp>(def)) {
        if (isa<arith::DivUIOp>(def)) {
          APInt divisor;
          if (!matchPattern(def->getOperand(1), m_ConstantInt(&divisor)) || !divisor.isStrictlyPositive()) return {};
        }
        for (Value operand : def->getOperands()) if (!clone(operand)) return {};
      } else return {};
      auto result = host.clone(*def, mapping)->getResult(0);
      check(result);
      return result;
    };
    Value computed = clone(bytes);
    if (!computed) {
      sizing.erase();
      return kernel.emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: dynamic size is not a supported launch-argument expression");
    }
    func::ReturnOp::create(host, kernel.getLoc(), computed);
    kernel->setAttr("tile.dynamic_shared_size", FlatSymbolRefAttr::get(host.getContext(), name));
    module.walk([&](gpu::LaunchFuncOp launch) {
      auto target = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(launch, launch.getKernelAttr());
      if (target != kernel) return;
      host.setInsertionPoint(launch);
      auto call = func::CallOp::create(host, launch.getLoc(), sizing, launch.getKernelOperands());
      auto zero = arith::ConstantIndexOp::create(host, launch.getLoc(), 0);
      auto valid = arith::CmpIOp::create(host, launch.getLoc(), arith::CmpIPredicate::sge, call.getResult(0), zero);
      cf::AssertOp::create(host, launch.getLoc(), valid,
                          "TILE_BARRIER_REUSE_MISSING_BARRIER: dynamic launch size exceeds nonnegative i32 range");
      auto count = arith::IndexCastOp::create(host, launch.getLoc(), host.getI32Type(), call.getResult(0));
      if (Value supplied = launch.getDynamicSharedMemorySize()) {
        auto equal = arith::CmpIOp::create(host, launch.getLoc(), arith::CmpIPredicate::eq, supplied, count);
        cf::AssertOp::create(host, launch.getLoc(), equal,
                            "TILE_BARRIER_REUSE_MISSING_BARRIER: explicit dynamic launch bytes disagree with arena");
      }
      launch.getDynamicSharedMemorySizeMutable().assign(count);
    });
    return success();
  }

  // Runtime-sized shared/LDS arena for one dominance cohort. Reuse groups keep
  // the maximum member size, just like the static planner; offsets are runtime
  // index expressions with natural alignment. The caller chooses the earliest
  // legal insertion point, so descriptors created in nested regions can own a
  // scoped arena rather than being illegally hoisted to function entry.
  void materializeDynamicSharedArena(
      Operation *func, const SmallVector<Operation *> &allocs,
      Operation *insertionPoint, OpBuilder &b) {
    if (!func || func->getRegion(0).empty() || allocs.empty() || !insertionPoint)
      return;
    // byteSize below needs a scalar element width. Decide that for every member
    // before any IR is built, so an unsupported element type leaves the cohort
    // unplaced instead of aborting midway through a partly-emitted arena.
    for (Operation *op : allocs) {
      auto type = dyn_cast<MemRefType>(op->getOperand(0).getType());
      if (!type || !type.getElementType().isIntOrFloat())
        return;
    }
    Location loc = insertionPoint->getLoc();
    b.setInsertionPoint(insertionPoint);
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    Value one = arith::ConstantIndexOp::create(b, loc, 1);

    IRMapping hoisted;
    if (auto kernel = dyn_cast<gpu::GPUFuncOp>(func))
      for (Value arg : kernel.getArguments()) hoisted.map(arg, arg);
    std::function<Value(Value)> hoistSize = [&](Value value) -> Value {
      if (hoisted.contains(value)) return hoisted.lookup(value);
      auto *def = value.getDefiningOp();
      if (!def || !value.getType().isIndex()) return {};
      if (auto dim = dyn_cast<memref::DimOp>(def)) {
        if (!hoisted.contains(dim.getSource()) || !hoistSize(dim.getIndex())) return {};
      } else if (isa<arith::ConstantIndexOp>(def)) {
      } else if (isa<arith::AddIOp, arith::MulIOp, arith::MaxUIOp, arith::DivUIOp>(def)) {
        for (Value operand : def->getOperands()) if (!hoistSize(operand)) return {};
      } else return {};
      return b.clone(*def, hoisted)->getResult(0);
    };

    auto byteSize = [&](Value value) {
      auto type = cast<MemRefType>(value.getType());
      Value elements = one;
      for (auto [index, extent] : llvm::enumerate(type.getShape())) {
        Value dim;
        if (extent == ShapedType::kDynamic)
          dim = b.createOrFold<memref::DimOp>(loc, value, index);
        else
          dim = arith::ConstantIndexOp::create(b, loc, extent);
        if (isa<gpu::GPUFuncOp>(func)) {
          dim = hoistSize(dim);
          if (!dim) return Value();
        }
        elements = arith::MulIOp::create(b, loc, elements, dim);
      }
      int64_t bits = type.getElementType().getIntOrFloatBitWidth();
      Value elementBytes =
          arith::ConstantIndexOp::create(b, loc, std::max<int64_t>((bits + 7) / 8, 1));
      return arith::MulIOp::create(b, loc, elements, elementBytes).getResult();
    };

    llvm::DenseMap<int64_t, Value> groupBytes;
    llvm::DenseMap<int64_t, int64_t> groupAlign;
    SmallVector<int64_t> order;
    for (Operation *op : allocs) {
      int64_t group =
          op->getAttrOfType<IntegerAttr>(kGroupAttr).getInt();
      Value size = byteSize(op->getOperand(0));
      if (!size) {
        func->emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: dynamic size is not a supported launch-argument expression");
        signalPassFailure();
        return;
      }
      auto found = groupBytes.find(group);
      if (found == groupBytes.end()) {
        groupBytes[group] = size;
        groupAlign[group] = elementAlign(op->getOperand(0));
        order.push_back(group);
      } else {
        found->second =
            arith::MaxUIOp::create(b, loc, found->second, size).getResult();
        groupAlign[group] =
            std::max(groupAlign[group], elementAlign(op->getOperand(0)));
      }
    }
    llvm::sort(order);

    llvm::DenseMap<int64_t, Value> offsets;
    Value cursor = zero;
    for (int64_t group : order) {
      int64_t align = std::max<int64_t>(groupAlign[group], 1);
      if (align > 1) {
        Value alignValue = arith::ConstantIndexOp::create(b, loc, align);
        Value alignMinusOne =
            arith::ConstantIndexOp::create(b, loc, align - 1);
        cursor = arith::AddIOp::create(b, loc, cursor, alignMinusOne);
        cursor = arith::DivUIOp::create(b, loc, cursor, alignValue);
        cursor = arith::MulIOp::create(b, loc, cursor, alignValue);
      }
      offsets[group] = cursor;
      cursor = arith::AddIOp::create(b, loc, cursor, groupBytes[group]);
    }

    Attribute memorySpace = b.getI64IntegerAttr(3);
    Value arena;
    if (auto kernel = dyn_cast<gpu::GPUFuncOp>(func)) {
      if (failed(materializeLaunchSizer(kernel, cursor))) {
        signalPassFailure();
        return;
      }
      memorySpace = gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup);
      auto arenaType = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(),
                                      MemRefLayoutAttrInterface(), memorySpace);
      arena = gpu::DynamicSharedMemoryOp::create(b, loc, arenaType).getResult();
    } else {
      auto arenaType = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(),
                                      MemRefLayoutAttrInterface(), memorySpace);
      arena = memref::AllocaOp::create(b, loc, arenaType, ValueRange{cursor},
                                     ValueRange{}, b.getI64IntegerAttr(16));
    }
    DominanceInfo dominance(func);

    for (Operation *alloc : allocs) {
      auto originalType = dyn_cast<MemRefType>(alloc->getOperand(0).getType());
      if (!originalType)
        continue;
      int64_t group =
          alloc->getAttrOfType<IntegerAttr>(kGroupAttr).getInt();
      auto viewType = MemRefType::get(
          originalType.getShape(), originalType.getElementType(),
          originalType.getLayout(), memorySpace);
      b.setInsertionPoint(alloc);
      SmallVector<Value> dynamicSizes;
      for (auto [index, extent] : llvm::enumerate(originalType.getShape()))
        if (extent == ShapedType::kDynamic)
          dynamicSizes.push_back(
              b.createOrFold<memref::DimOp>(alloc->getLoc(),
                                           alloc->getOperand(0), index));
      OperationState viewState(alloc->getLoc(), "memref.view");
      viewState.addOperands(arena);
      viewState.addOperands(offsets[group]);
      viewState.addOperands(dynamicSizes);
      viewState.addTypes(viewType);
      Operation *view = b.create(viewState);
      Value original = alloc->getOperand(0);
      retargetAliases(original, memorySpace);
      for (OpOperand &use :
           llvm::make_early_inc_range(original.getUses())) {
        if (use.getOwner() != alloc &&
            dominance.properlyDominates(alloc, use.getOwner()))
          use.set(view->getResult(0));
      }
      alloc->erase();
    }
    func->setAttr("tile.smem_arena_dynamic", b.getUnitAttr());
    func->setAttr("tile.smem_arena_materialized", b.getUnitAttr());
  }

  // Realize the offset plan as one address-space-3 workgroup global plus typed
  // memref.view slices. A memref.alloca would lower to llvm.alloca even with an
  // address-space-3 pointer; AMDGPU does not account that object as statically
  // reserved LDS. A module-level memref.global lowers to the real addrspace(3)
  // workgroup object consumed by both ROCDL and NVPTX resource accounting.
  void materializeSharedArena(Operation *fn,
                              const SmallVector<Operation *> &allocs,
                              int64_t arenaBytes, OpBuilder &b) {
    if (fn->getNumRegions() != 1 || fn->getRegion(0).empty())
      return;
    Location loc = fn->getLoc();
    auto memorySpace = b.getI64IntegerAttr(3);
    auto arenaType = MemRefType::get(
        {arenaBytes}, b.getI8Type(), MemRefLayoutAttrInterface(), memorySpace);
    auto functionName = SymbolTable::getSymbolName(fn).getValue();
    std::string arenaName =
        ("__tessera_smem_arena_" + functionName).str();
    b.setInsertionPoint(fn);
    memref::GlobalOp::create(
        b, loc, arenaName, b.getStringAttr("private"), arenaType,
        b.getUnitAttr(), false, b.getI64IntegerAttr(16));
    b.setInsertionPointToStart(&fn->getRegion(0).front());
    auto arena =
        memref::GetGlobalOp::create(b, loc, arenaType, arenaName);
    DominanceInfo dominance(fn);

    for (Operation *alloc : allocs) {
      auto offset = alloc->getAttrOfType<IntegerAttr>("tile.smem_offset");
      auto originalType =
          dyn_cast<MemRefType>(alloc->getOperand(0).getType());
      if (!offset || !originalType)
        continue;
      auto viewType = MemRefType::get(
          originalType.getShape(), originalType.getElementType(),
          originalType.getLayout(), memorySpace);
      b.setInsertionPoint(alloc);
      Value byteShift = arith::ConstantIndexOp::create(
          b, alloc->getLoc(), offset.getInt());
      OperationState viewState(alloc->getLoc(), "memref.view");
      viewState.addOperands({arena.getResult(), byteShift});
      viewState.addTypes(viewType);
      Operation *view = b.create(viewState);
      Value original = alloc->getOperand(0);
      retargetAliases(original, memorySpace);
      for (OpOperand &use :
           llvm::make_early_inc_range(original.getUses())) {
        if (use.getOwner() != alloc &&
            dominance.properlyDominates(alloc, use.getOwner()))
          use.set(view->getResult(0));
      }
      alloc->erase();
    }
    fn->setAttr("tile.smem_arena_materialized", b.getUnitAttr());
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createTileBufferArenaPass() {
  return std::make_unique<TileBufferArena>();
}
}  // namespace tessera
