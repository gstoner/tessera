// TileBufferReusePass.cpp — global buffer assignment/reuse for Tile IR
// (Workstream H / W3, 2026-07-08). The *assignment* half of shared-memory
// planning, paired with TileBarrierReuseLegalityPass as its correctness verifier
// (the same two-sided pattern as LayoutAssignmentPass ↔ LayoutLegalityPass).
//
// Tiled GEMM / attention kernels stage operands through `tile.alloc_shared` (LDS)
// and `tile.tmem.alloc` (Blackwell TMEM) buffers. When two such buffers have
// **disjoint live ranges**, they can share one physical backing — cutting peak
// shared-memory footprint, which directly gates occupancy. This pass computes a
// conservative per-buffer live range (first-to-last reference in program order),
// greedily colors buffers of identical memref type into reuse groups (a classic
// interval-coloring / left-edge assignment), and stamps the group on each alloc:
//
//   tile.alloc_shared %buf {tile.buffer_group = N} : memref<...>
//
// It also records the static footprint saved as function attributes
// (`tile.buffer_reuse.bytes_before/after/groups`). Correctness is by construction
// — only NON-overlapping live ranges share a group, so no live buffer is ever
// clobbered. Like LayoutAssignmentPass v1 the output is IR metadata: a
// shared-memory-aware backend reads `tile.buffer_group` to alias the physical
// allocation; hardware-free here so it is lit-testable before any emission.

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
constexpr StringRef kBytesBefore = "tile.buffer_reuse.bytes_before";
constexpr StringRef kBytesAfter = "tile.buffer_reuse.bytes_after";
constexpr StringRef kGroups = "tile.buffer_reuse.groups";

// Tile-IR allocation ops whose buffer this pass plans. The buffer's SSA value is
// the memref operand (operand 0 in both ops' ODS).
static bool isAllocOp(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return n == "tile.alloc_shared" || n == "tile.tmem.alloc";
}

// Static byte size of a memref value, or -1 when it is not statically known (a
// dynamic dim / non-memref) — such a buffer never joins a reuse group.
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

struct Buffer {
  Operation *alloc;   // the alloc op to stamp
  Value memref;       // the buffer's SSA value
  int64_t start;      // first program index that references it (the alloc)
  int64_t end;        // last program index that references it
  int64_t bytes;      // static size, or -1 if unknown
  int group = -1;
};

struct TileBufferReuse
    : public PassWrapper<TileBufferReuse, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBufferReuse)

  StringRef getArgument() const override { return "tessera-tile-buffer-reuse"; }
  StringRef getDescription() const override {
    return "Global buffer assignment/reuse for Tile IR — assign disjoint-live-"
           "range tile.alloc_shared / tile.tmem.alloc buffers of identical type "
           "to shared reuse groups (tile.buffer_group), cutting peak shared "
           "memory. The assignment half of shared-memory planning; "
           "TileBarrierReuseLegalityPass verifies it.";
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *fn) {
      if (fn->getName().getStringRef() == "func.func" && fn->getNumRegions())
        planRegion(fn);
    });
  }

  // Plan one function body: index ops, derive live ranges, color, stamp.
  void planRegion(Operation *fn) {
    llvm::DenseMap<Operation *, int64_t> index;
    int64_t next = 0;
    fn->walk([&](Operation *op) { index[op] = next++; });

    SmallVector<Buffer> buffers;
    fn->walk([&](Operation *op) {
      if (!isAllocOp(op) || op->getNumOperands() == 0)
        return;
      Value buf = op->getOperand(0);
      if (!isa<MemRefType>(buf.getType()))
        return;
      int64_t start = index[op], end = index[op];
      for (Operation *user : buf.getUsers())
        end = std::max(end, index.lookup(user));
      buffers.push_back({op, buf, start, end, staticByteSize(buf), -1});
    });
    if (buffers.empty())
      return;

    // Left-edge greedy coloring: process by ascending start; a buffer joins the
    // first group whose last member's live range ended strictly before this one
    // begins AND whose memref type matches exactly (same backing size/layout).
    SmallVector<unsigned> order(llvm::to_vector(llvm::seq<unsigned>(
        0, buffers.size())));
    llvm::sort(order, [&](unsigned a, unsigned b) {
      if (buffers[a].start != buffers[b].start)
        return buffers[a].start < buffers[b].start;
      return buffers[a].end < buffers[b].end;
    });

    struct Group {
      int64_t lastEnd;
      Type type;
      int64_t bytes;
    };
    SmallVector<Group> groups;
    for (unsigned i : order) {
      Buffer &b = buffers[i];
      int chosen = -1;
      // A buffer of unknown static size is never aliased (own group).
      if (b.bytes >= 0) {
        for (unsigned g = 0; g < groups.size(); ++g) {
          if (groups[g].lastEnd < b.start && groups[g].type == b.memref.getType()) {
            chosen = g;
            break;
          }
        }
      }
      if (chosen < 0) {
        chosen = groups.size();
        groups.push_back({b.end, b.memref.getType(), b.bytes});
      } else {
        groups[chosen].lastEnd = b.end;
      }
      b.group = chosen;
    }

    // Stamp the group on each alloc; tally static footprint before/after.
    OpBuilder builder(fn->getContext());
    int64_t bytesBefore = 0;
    SmallVector<int64_t> groupBytes(groups.size(), 0);
    for (const Buffer &b : buffers) {
      b.alloc->setAttr(kGroupAttr,
                       builder.getI64IntegerAttr(b.group));
      if (b.bytes >= 0) {
        bytesBefore += b.bytes;
        groupBytes[b.group] = std::max(groupBytes[b.group], b.bytes);
      }
    }
    int64_t bytesAfter = 0;
    for (int64_t gb : groupBytes)
      bytesAfter += gb;

    fn->setAttr(kBytesBefore, builder.getI64IntegerAttr(bytesBefore));
    fn->setAttr(kBytesAfter, builder.getI64IntegerAttr(bytesAfter));
    fn->setAttr(kGroups, builder.getI64IntegerAttr((int64_t)groups.size()));
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createTileBufferReusePass() {
  return std::make_unique<TileBufferReuse>();
}
}  // namespace tessera
