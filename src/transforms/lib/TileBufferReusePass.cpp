// TileBufferReusePass.cpp — global buffer assignment/reuse for Tile IR
// (Workstream H / W3, 2026-07-08). The *assignment* half of shared-memory
// planning, paired with TileBarrierReuseLegalityPass as its correctness verifier
// (the same two-sided pattern as LayoutAssignmentPass ↔ LayoutLegalityPass).
//
// Tiled GEMM / attention kernels stage operands through `tile.alloc_shared` (LDS)
// and `tile.tmem.alloc` (Blackwell TMEM) buffers. When two such buffers have
// **disjoint live ranges**, they can share one physical backing — cutting peak
// shared-memory footprint, which directly gates occupancy. This pass computes a
// conservative alias-inclusive live range (including async completion),
// greedily colors buffers of identical memref type into reuse groups (a classic
// interval-coloring / left-edge assignment), and stamps the group on each alloc:
//
//   tile.alloc_shared %buf {tile.buffer_group = N} : memref<...>
//
// It also records the static footprint saved as function attributes
// (`tile.buffer_reuse.bytes_before/after/groups`). Correctness is by construction
// — only proven NON-overlapping live ranges share a group. TileBufferArenaPass
// rechecks the same proof before physically materializing workgroup storage.

#include "Tessera/Transforms/Passes.h"
#include "TileMemrefLifetime.h"

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
  return tessera::memory::staticBytes(v);
}

struct Buffer {
  Operation *alloc;   // the alloc op to stamp
  Value memref;       // the buffer's SSA value
  StringRef kind;     // the alloc op name — SMEM (alloc_shared) vs TMEM never mix
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
           "TileBufferArenaPass rechecks the shared lifetime proof.";
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *fn) {
      if (fn->getName().getStringRef() == "func.func" && fn->getNumRegions())
        planRegion(fn);
    });
  }

  // Plan one function body: index ops, derive live ranges, color, stamp.
  void planRegion(Operation *fn) {
    tessera::memory::Lifetimes lifetimes(fn);

    SmallVector<Buffer> buffers;
    fn->walk([&](Operation *op) {
      if (!isAllocOp(op) || op->getNumOperands() == 0)
        return;
      Value buf = op->getOperand(0);
      if (!isa<MemRefType>(buf.getType()))
        return;
      auto live = lifetimes.get(op);
      int64_t start = live.start, end = live.end;
      buffers.push_back({op, buf, op->getName().getStringRef(), start, end,
                         staticByteSize(buf), -1});
    });
    if (buffers.empty())
      return;

    // Deterministic interference coloring: every existing group member must
    // prove noninterference, including exclusive structured paths. Comparing
    // only the last member is insufficient once intervals span branches.
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
      StringRef kind;
      int64_t bytes;
      SmallVector<Operation *> members;
    };
    SmallVector<Group> groups;
    for (unsigned i : order) {
      Buffer &b = buffers[i];
      int chosen = -1;
      // A buffer of unknown static size is never aliased (own group). Two buffers
      // share a group only if their live ranges are disjoint AND they are the same
      // alloc kind (SMEM `alloc_shared` vs TMEM `tmem.alloc` are distinct physical
      // spaces — a backend cannot realize one group as both) AND the same memref
      // type (identical backing size + element type + layout + memory space).
      if (b.bytes >= 0) {
        for (unsigned g = 0; g < groups.size(); ++g) {
          if (groups[g].kind == b.kind && groups[g].type == b.memref.getType() &&
              llvm::all_of(groups[g].members, [&](Operation *member) { return lifetimes.disjoint(member, b.alloc); })) {
            chosen = g;
            break;
          }
        }
      }
      if (chosen < 0) {
        chosen = groups.size();
        groups.push_back({b.end, b.memref.getType(), b.kind, b.bytes, {}});
      } else {
        groups[chosen].lastEnd = b.end;
      }
      groups[chosen].members.push_back(b.alloc);
      b.group = chosen;
    }

    // Stamp the group on each alloc; tally static footprint before/after.
    OpBuilder builder(fn->getContext());
    __int128 bytesBefore = 0;
    SmallVector<int64_t> groupBytes(groups.size(), 0);
    for (const Buffer &b : buffers) {
      b.alloc->setAttr(kGroupAttr,
                       builder.getI64IntegerAttr(b.group));
      if (b.bytes >= 0) {
        bytesBefore += b.bytes;
        groupBytes[b.group] = std::max(groupBytes[b.group], b.bytes);
      }
    }
    __int128 bytesAfter = 0;
    for (int64_t gb : groupBytes)
      bytesAfter += gb;

    fn->setAttr(kBytesBefore, builder.getI64IntegerAttr(bytesBefore <= std::numeric_limits<int64_t>::max() ? static_cast<int64_t>(bytesBefore) : -1));
    fn->setAttr(kBytesAfter, builder.getI64IntegerAttr(bytesAfter <= std::numeric_limits<int64_t>::max() ? static_cast<int64_t>(bytesAfter) : -1));
    fn->setAttr(kGroups, builder.getI64IntegerAttr((int64_t)groups.size()));
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createTileBufferReusePass() {
  return std::make_unique<TileBufferReuse>();
}
}  // namespace tessera
