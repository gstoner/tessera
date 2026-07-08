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
    // group id -> max static byte size (-1 if any member is unknown).
    llvm::DenseMap<int64_t, int64_t> groupBytes;
    SmallVector<int64_t> order;                 // ascending unique group ids
    for (Operation *op : allocs) {
      int64_t g = op->getAttrOfType<IntegerAttr>(kGroupAttr).getInt();
      int64_t sz = staticByteSize(op->getOperand(0));
      auto it = groupBytes.find(g);
      if (it == groupBytes.end()) {
        groupBytes[g] = sz;
        order.push_back(g);
      } else if (sz < 0 || it->second < 0) {
        it->second = -1;                        // group poisoned by an unknown dim
      } else {
        it->second = std::max(it->second, sz);
      }
    }
    llvm::sort(order);
    llvm::DenseMap<int64_t, int64_t> offset;
    int64_t cursor = 0;
    for (int64_t g : order) {
      if (groupBytes[g] < 0)
        continue;                               // unplaceable — leave unstamped
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
    int64_t smemBytes = layoutSpace(smem, "tile.smem_offset", b);
    int64_t tmemBytes = layoutSpace(tmem, "tile.tmem_offset", b);
    if (!smem.empty())
      fn->setAttr("tile.smem_arena_bytes", b.getI64IntegerAttr(smemBytes));
    if (!tmem.empty())
      fn->setAttr("tile.tmem_arena_bytes", b.getI64IntegerAttr(tmemBytes));
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createTileBufferArenaPass() {
  return std::make_unique<TileBufferArena>();
}
}  // namespace tessera
