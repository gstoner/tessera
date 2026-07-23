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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  auto mr = dyn_cast<MemRefType>(v.getType());
  if (!mr || !mr.hasStaticShape())
    return -1;
  int64_t elems = 1;
  for (int64_t d : mr.getShape())
    elems *= d;
  int64_t bits = mr.getElementType().getIntOrFloatBitWidth();
  if (bits <= 0)
    return -1;
  return elems * ((bits + 7) / 8);
}

// Natural alignment (bytes) of a memref's element — a backend casts
// `arena + offset` to `T*`, so each group's offset must be a multiple of this or
// the typed access is misaligned. Scalar alignment = element byte width.
static int64_t elementAlign(Value v) {
  auto mr = dyn_cast<MemRefType>(v.getType());
  if (!mr)
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
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *fn) {
      if (fn->getName().getStringRef() == "func.func" && fn->getNumRegions())
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
      cursor = ((cursor + a - 1) / a) * a;      // pad up to the group's alignment
      offset[g] = cursor;
      cursor += groupBytes[g];
    }
    for (Operation *op : allocs) {
      int64_t g = op->getAttrOfType<IntegerAttr>(kGroupAttr).getInt();
      auto it = offset.find(g);
      if (it != offset.end())
        op->setAttr(offsetAttr, b.getI64IntegerAttr(it->second));
    }
    return cursor;
  }

  void layoutRegion(Operation *fn) {
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
    if (!smem.empty() && !dynamicSmem)
      fn->setAttr("tile.smem_arena_bytes", b.getI64IntegerAttr(smemBytes));
    if (!tmem.empty())
      fn->setAttr("tile.tmem_arena_bytes", b.getI64IntegerAttr(tmemBytes));
    if (!smem.empty() && !dynamicSmem && smemBytes > 0)
      materializeSharedArena(fn, smem, smemBytes, b);
  }

  // Runtime-sized shared/LDS arena for one dominance cohort. Reuse groups keep
  // the maximum member size, just like the static planner; offsets are runtime
  // index expressions with natural alignment. The caller chooses the earliest
  // legal insertion point, so descriptors created in nested regions can own a
  // scoped arena rather than being illegally hoisted to function entry.
  void materializeDynamicSharedArena(
      func::FuncOp func, const SmallVector<Operation *> &allocs,
      Operation *insertionPoint, OpBuilder &b) {
    if (!func || func.empty() || allocs.empty() || !insertionPoint)
      return;
    Location loc = insertionPoint->getLoc();
    b.setInsertionPoint(insertionPoint);
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    Value one = arith::ConstantIndexOp::create(b, loc, 1);

    auto byteSize = [&](Value value) {
      auto type = cast<MemRefType>(value.getType());
      Value elements = one;
      for (auto [index, extent] : llvm::enumerate(type.getShape())) {
        Value dim;
        if (extent == ShapedType::kDynamic)
          dim = memref::DimOp::create(b, loc, value, index).getResult();
        else
          dim = arith::ConstantIndexOp::create(b, loc, extent);
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

    auto memorySpace = b.getI64IntegerAttr(3);
    auto arenaType = MemRefType::get(
        {ShapedType::kDynamic}, b.getI8Type(), MemRefLayoutAttrInterface(),
        memorySpace);
    auto arena = memref::AllocaOp::create(
        b, loc, arenaType, ValueRange{cursor}, ValueRange{},
        b.getI64IntegerAttr(16));
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
              memref::DimOp::create(b, alloc->getLoc(),
                                    alloc->getOperand(0), index).getResult());
      OperationState viewState(alloc->getLoc(), "memref.view");
      viewState.addOperands(arena.getResult());
      viewState.addOperands(offsets[group]);
      viewState.addOperands(dynamicSizes);
      viewState.addTypes(viewType);
      Operation *view = b.create(viewState);
      Value original = alloc->getOperand(0);
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
    auto func = dyn_cast<func::FuncOp>(fn);
    if (!func || func.empty())
      return;
    Location loc = func.getLoc();
    auto memorySpace = b.getI64IntegerAttr(3);
    auto arenaType = MemRefType::get(
        {arenaBytes}, b.getI8Type(), MemRefLayoutAttrInterface(), memorySpace);
    auto functionName = func.getSymName();
    std::string arenaName =
        ("__tessera_smem_arena_" + functionName).str();
    b.setInsertionPoint(fn);
    memref::GlobalOp::create(
        b, loc, arenaName, b.getStringAttr("private"), arenaType,
        b.getUnitAttr(), false, b.getI64IntegerAttr(16));
    b.setInsertionPointToStart(&func.front());
    auto arena =
        memref::GetGlobalOp::create(b, loc, arenaType, arenaName);
    DominanceInfo dominance(func);

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
