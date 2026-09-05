// TileBarrierReuseLegalityPass.cpp — C2 (2026-06-23)
//
// "Barriers are a layout-reuse correctness property, not a scheduling artifact."
// (TIRx review / COMPILER_AUDIT item C2.) The motivating case is FA-4's TMEM
// allocation aliased as an fp32 view (S/O) and an fp16 view (P): the barriers
// exist because each region is *reused* strictly after its prior consumer
// finishes. This pass turns that into a checkable rule on Tile IR carrying the
// C1 `#tile.layout` attribute:
//
//   For a given buffer, if two WRITE ops target overlapping STORAGE-axis
//   (m / tlane / tcol) footprints of their `#tile.layout` with NO intervening
//   barrier op, emit TILE_BARRIER_REUSE_MISSING_BARRIER on the second writer.
//
// This is LayoutLegalityPass's sibling — a finite may-live analysis and stable diagnostic code,
// registered standalone as `--tessera-tile-barrier-reuse-legality`. It is the
// forcing function / acceptance gate for the typed-barrier + reuse work (C3):
// once WarpSpecialization emits real barriers, "does this pass go green on the
// FA-4 fixture?" becomes the correctness check.
//
// The local write/write check uses !tile.buffer allocation identity plus
// tile.layout footprints. Only registered completing waits/barriers clear a
// pending hazard; policy attributes and nonblocking polls do not. Allocation-
// scoped release and structured/CFG lifetime joins are derived from SSA.

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"
#include "TileRelationalLegality.h"
#include "Tessera/Dialect/Tile/TileSSAOrigins.h"
#include "tessera/ProgrammingModel/ScheduleDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <set>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <limits>
#include <utility>

using namespace mlir;

namespace {

// Storage axes — the ones that name a physical memory/TMEM location and can
// therefore alias. Placement on register/lane/warp/grid axes does not alias a
// shared storage region.
static bool isStorageAxis(StringRef ax) {
  // Memory-resident axes that can alias: `m` (linear / NVIDIA SMEM), `lds` (AMD
  // Local Data Share), `tlane`/`tcol` (NVIDIA TMEM). LDS reuse needs an
  // intervening barrier exactly as SMEM does — ROCm is first-class here.
  return ax == "m" || ax == "lds" || ax == "tlane" || ax == "tcol";
}

// Linear footprint of a layout restricted to its storage-axis shard dims:
// Include both ends of signed-stride shards, using wide intermediates. Returns
// nullopt when the layout touches no storage axis (a pure register/lane
// fragment — no shared-storage hazard).
static std::optional<std::pair<int64_t, int64_t>>
storageFootprint(tessera::tile::TileLayoutAttr layout) {
  ArrayRef<int64_t> extents = layout.getShardExtents();
  ArrayRef<int64_t> strides = layout.getShardStrides();
  ArrayRef<StringAttr> axes = layout.getShardAxes();
  __int128 lo = layout.getOffset(), hi = lo + 1;
  bool anyStorage = false;
  for (auto [extent, stride, ax] : llvm::zip(extents, strides, axes)) {
    if (!isStorageAxis(ax.getValue())) continue;
    anyStorage = true;
    __int128 delta = (static_cast<__int128>(extent) - 1) * stride;
    if (delta < 0) lo += delta;
    else hi += delta;
  }
  if (!anyStorage) return std::nullopt;
  // A footprint outside signed index range cannot prove disjoint storage.
  if (lo < std::numeric_limits<int64_t>::min() || hi > std::numeric_limits<int64_t>::max())
    return std::make_pair(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
  return std::make_pair(static_cast<int64_t>(lo), static_cast<int64_t>(hi));
}

static bool overlaps(const std::pair<int64_t, int64_t> &a,
                     const std::pair<int64_t, int64_t> &b) {
  return a.first < b.second && b.first < a.second;
}

// A may-live set retains every access, including disjoint earlier footprints.
// Completion kills the producing access, never all uses of the same allocation.
struct Access {
  Operation *op;
  Value root;
  std::optional<std::pair<int64_t, int64_t>> footprint;
  SmallVector<Value> completions;
  bool operator==(const Access &other) const {
    return op == other.op && root == other.root && footprint == other.footprint &&
           completions.size() == other.completions.size() &&
           llvm::all_of(completions, [&](Value v) { return llvm::is_contained(other.completions, v); });
  }
};
struct LifetimeState {
  SmallVector<Access> pending;
  llvm::SmallPtrSet<Value, 8> freed;
  bool merge(const LifetimeState &other) {
    bool changed = false;
    for (const auto &access : other.pending)
      if (!llvm::is_contained(pending, access)) {
        pending.push_back(access);
        changed = true;
      }
    for (Value root : other.freed)
      changed |= freed.insert(root).second;
    return changed;
  }
};

struct TileBarrierReuseLegality
    : public PassWrapper<TileBarrierReuseLegality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBarrierReuseLegality)

  StringRef getArgument() const override {
    return "tessera-tile-barrier-reuse-legality";
  }
  StringRef getDescription() const override {
    return "Tile allocation lifetime legality: scoped completion, storage "
           "reuse, deallocation, and may-live branch/loop/CFG joins.";
  }

  bool anyError = false;
  std::set<std::pair<Operation *, Operation *>> reported;

  void report(Operation *op, Operation *previous, StringRef reason) {
    if (!reported.insert({op, previous}).second)
      return;
    anyError = true;
    auto diag = op->emitOpError("TILE_BARRIER_REUSE_MISSING_BARRIER: buffer SSA allocation ");
    diag << reason;
    if (previous)
      diag.attachNote(previous->getLoc())
          << "previous write to the same allocation root here (or outstanding async access)";
  }

  void roots(Value value, SmallVectorImpl<Value> &out, Operation *use) {
    auto origins = tessera::tile::resolveSSAOrigins(value);
    if (!origins.complete) report(use, nullptr, "has unresolved allocation provenance");
    for (Value root : origins.roots) {
      if (!root.getDefiningOp<tessera::tile::AllocOp>())
        report(use, nullptr, "has unresolved allocation provenance");
      else if (!llvm::is_contained(out, root)) out.push_back(root);
    }
  }

  // Rename completion capabilities at an SSA edge. Dropping local names is
  // essential: executing the same copy operation again creates a NEW event.
  void renameTokens(LifetimeState &state, ValueRange from, ValueRange to,
                    Region *expired = nullptr) {
    for (Access &access : state.pending) {
      SmallVector<Value> names;
      for (auto [source, target] : llvm::zip(from, to))
        if (isa<tessera::tile::AsyncTokenType>(target.getType()) &&
            llvm::is_contained(access.completions, source) &&
            !llvm::is_contained(names, target)) names.push_back(target);
      for (Value name : access.completions) {
        Region *owner = name.getParentRegion();
        if (expired && (owner == expired || expired->isAncestor(owner))) continue;
        if (!llvm::is_contained(names, name)) names.push_back(name);
      }
      access.completions = std::move(names);
    }
  }

  LifetimeState branchResult(Region &region, const LifetimeState &entry,
                             ValueRange results) {
    auto state = analyze(region, entry);
    if (!region.empty())
      renameTokens(state, region.front().getTerminator()->getOperands(), results, &region);
    return state;
  }

  void transfer(Operation *op, LifetimeState &state) {
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      llvm::APInt condition;
      if (matchPattern(branch.getCondition(), m_ConstantInt(&condition))) {
        Region &selected = condition.isZero() ? branch.getElseRegion() : branch.getThenRegion();
        if (!selected.empty()) state = branchResult(selected, state, branch.getResults());
        return;
      }
      auto thenState = branchResult(branch.getThenRegion(), state, branch.getResults());
      auto elseState = branch.getElseRegion().empty()
                           ? state : branchResult(branch.getElseRegion(), state, branch.getResults());
      thenState.merge(elseState);
      state = std::move(thenState);
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto initial = state;
      renameTokens(initial, loop.getInitArgs(), loop.getRegionIterArgs());
      auto zeroTrip = state;
      renameTokens(zeroTrip, loop.getInitArgs(), loop.getResults());
      llvm::APInt lower, upper, step;
      bool knownNonzero = false, singleTrip = false;
      if (matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
          matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) &&
          matchPattern(loop.getStep(), m_ConstantInt(&step)) && step.isStrictlyPositive()) {
        int64_t lo = lower.getSExtValue(), hi = upper.getSExtValue();
        if (lo >= hi) { state = std::move(zeroTrip); return; }
        knownNonzero = true;
        singleTrip = static_cast<__int128>(hi) - lo <= step.getSExtValue();
      }
      auto header = initial;
      LifetimeState exits;
      if (!knownNonzero) exits.merge(zeroTrip);
      while (true) {
        auto body = analyze(loop.getRegion(), header);
        auto exit = body;
        auto yielded = loop.getBody()->getTerminator()->getOperands();
        renameTokens(exit, yielded, loop.getResults(), &loop.getRegion());
        exits.merge(exit);
        if (singleTrip) break;
        renameTokens(body, yielded, loop.getRegionIterArgs(), &loop.getRegion());
        if (!header.merge(body)) break;
      }
      state = std::move(exits);
      return;
    }
    if (isa<tessera::schedule::MeshRegionOp>(op)) {
      state = analyze(op->getRegion(0), state);
      return;
    }
    if (isa<tessera::schedule::WarpOp>(op)) {
      // A role-local rendezvous cannot retire accesses inherited from another
      // role. Retain them across this region; cross-role completion needs an
      // explicit ownership/token-generation proof rather than lexical order.
      auto inherited = state;
      state = analyze(op->getRegion(0), state);
      state.merge(inherited);
      return;
    }
    if (op->getNumRegions()) {
      bool relevant = false;
      op->walk([&](Operation *nested) {
        relevant |= llvm::any_of(nested->getOperandTypes(), [](Type t) {
          return isa<tessera::tile::BufferType>(t);
        });
      });
      if (relevant) report(op, nullptr, "uses an unsupported region lifetime");
      return;
    }
    if (auto alloc = dyn_cast<tessera::tile::AllocOp>(op)) {
      // A region-local allocation is fresh on each dynamic execution.
      Value root = alloc->getResult(0);
      llvm::erase_if(state.pending, [&](const Access &a) { return a.root == root; });
      state.freed.erase(root);
      return;
    }
    if (isa<tessera::tile::CtaSyncOp, tessera::tile::SBarrierOp>(op)) {
      // Thread rendezvous does not itself complete outstanding DMA.
      llvm::erase_if(state.pending, [](const Access &a) {
        return !isa<tessera::tile::AsyncCopyOp, tessera::tile::TMACopyAsyncOp>(a.op);
      });
      return;
    }
    if (isa<tessera::tile::WaitAsyncOp, tessera::tile::MBarrierWaitOp>(op)) {
      llvm::SmallPtrSet<Value, 4> completed;
      bool typed = false;
      for (Value operand : op->getOperands()) {
        if (!isa<tessera::tile::AsyncTokenType>(operand.getType())) continue;
        typed = true;
        completed.insert(operand);
      }
      auto stage = op->getAttr("stage");
      auto barrier = op->getAttr("tile.barrier_id");
      if (!typed && isa<tessera::tile::WaitAsyncOp>(op) && !stage && !barrier) {
        state.pending.clear(); // declared legacy wait-all contract
      } else {
        llvm::erase_if(state.pending, [&](const Access &a) {
          if (typed) return llvm::any_of(a.completions, [&](Value token) { return completed.contains(token); });
          if (!isa<tessera::tile::WaitAsyncOp>(op) ||
              !isa<tessera::tile::AsyncCopyOp>(a.op)) return false;
          return (!stage || a.op->getAttr("stage") == stage) &&
                 (!barrier || a.op->getAttr("tile.barrier_id") == barrier);
        });
      }
      // An arrival token alone proves transaction count, not buffer ownership.
      return;
    }
    SmallVector<Value> buffers;
    for (Value value : op->getOperands())
      if (isa<tessera::tile::BufferType>(value.getType())) roots(value, buffers, op);
    bool markerWrite = isa<tessera::tile::BufferWriteOp>(op);
    bool mmaWrite = isa<tessera::tile::MMAOp>(op) && buffers.size() == 1;
    bool write = markerWrite || mmaWrite;
    bool async = isa<tessera::tile::AsyncCopyOp, tessera::tile::TMACopyAsyncOp>(op);
    if (auto copy = dyn_cast<tessera::tile::TMACopyAsyncOp>(op)) {
      auto origins = tessera::tile::resolveSSAOrigins(copy.getDescriptor());
      if (!origins.complete)
        report(op, nullptr, "has an unresolved TMA descriptor lifetime");
      for (Value origin : origins.roots) {
        auto descriptor = origin.getDefiningOp<tessera::tile::TMADescriptorOp>();
        if (!descriptor) {
          report(op, nullptr, "has an opaque TMA descriptor lifetime");
          continue;
        }
        if (isa<tessera::tile::BufferType>(descriptor.getSource().getType()))
          roots(descriptor.getSource(), buffers, op);
      }
    }
    bool dealloc = isa<tessera::tile::DeallocOp>(op);
    SmallVector<Value> destinations;
    if (markerWrite) roots(cast<tessera::tile::BufferWriteOp>(op).getBuffer(), destinations, op);
    if (mmaWrite) destinations.append(buffers.begin(), buffers.end());
    if (isa<tessera::tile::AsyncCopyOp>(op) && op->getNumOperands() >= 2 &&
        isa<tessera::tile::BufferType>(op->getOperand(0).getType()) &&
        isa<tessera::tile::BufferType>(op->getOperand(1).getType()))
      roots(op->getOperand(0), destinations, op);
    else if (isa<tessera::tile::AsyncCopyOp>(op) && op->getNumOperands() >= 2 &&
             isa<RankedTensorType>(op->getOperand(0).getType()) && buffers.size() == 1)
      destinations.append(buffers.begin(), buffers.end());
    if (!buffers.empty() && !write && !async && !dealloc &&
        !isa<tessera::tile::TMADescriptorOp>(op) && !op->hasTrait<OpTrait::IsTerminator>())
      report(op, nullptr, "has an unsupported buffer-access lifetime");
    for (Value root : buffers) {
      if (state.freed.contains(root))
        report(op, nullptr, "is used after a possible deallocation");
      auto layout = op->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
      auto footprint = layout ? storageFootprint(layout) : std::nullopt;
      bool destination = llvm::is_contained(destinations, root);
      if (destination || dealloc) {
        for (const Access &previous : state.pending)
          if (previous.root == root &&
              (dealloc || !footprint || !previous.footprint ||
               overlaps(*footprint, *previous.footprint)))
            report(op, previous.op, dealloc ? "is deallocated before completion"
                                          : "is reused before completion");
      }
      if (dealloc) {
        state.freed.insert(root);
        llvm::erase_if(state.pending, [&](const Access &a) { return a.root == root; });
      } else if ((write && destination && (!layout || footprint)) || async) {
        Access access{op, root, footprint, {}};
        if (async)
          for (Value result : op->getResults())
            if (isa<tessera::tile::AsyncTokenType>(result.getType())) access.completions.push_back(result);
        if (!llvm::is_contained(state.pending, access)) state.pending.push_back(access);
      }
    }
  }

  LifetimeState analyze(Region &region, const LifetimeState &initial) {
    if (region.empty()) return initial;
    llvm::DenseMap<Block *, LifetimeState> incoming;
    llvm::SmallPtrSet<Block *, 8> reached;
    SmallVector<Block *> worklist{&region.front()};
    incoming[&region.front()] = initial;
    reached.insert(&region.front());
    LifetimeState exits;
    while (!worklist.empty()) {
      Block *block = worklist.pop_back_val();
      auto state = incoming[block];
      for (Operation &op : *block) transfer(&op, state);
      Operation *terminator = block->getTerminator();
      if (terminator->getNumSuccessors() == 0) exits.merge(state);
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        Block *successor = terminator->getSuccessor(i);
        auto edge = state;
        if (auto branch = dyn_cast<BranchOpInterface>(terminator)) {
          auto operands = branch.getSuccessorOperands(i);
          SmallVector<Value> from, to;
          for (unsigned j = 0; j < successor->getNumArguments(); ++j)
            if (Value source = operands[j]) { from.push_back(source); to.push_back(successor->getArgument(j)); }
          // Rebinding a block-local result/argument must retire its old name.
          for (Access &access : edge.pending) {
            SmallVector<Value> mapped;
            for (auto [source, target] : llvm::zip(from, to))
              if (llvm::is_contained(access.completions, source)) mapped.push_back(target);
            llvm::erase_if(access.completions, [&](Value v) { return v.getParentBlock() == successor; });
            for (Value v : mapped)
              if (!llvm::is_contained(access.completions, v)) access.completions.push_back(v);
          }
        }
        bool changed = incoming[successor].merge(edge);
        if (reached.insert(successor).second || changed) worklist.push_back(successor);
      }
    }
    return exits;
  }

  LogicalResult verify(ModuleOp module) {
    module.walk([&](func::FuncOp func) { analyze(func.getBody(), {}); });
    return failure(anyError);
  }

  void runOnOperation() override {
    if (failed(verify(getOperation())))
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
LogicalResult verifyTileBarrierReuseRelations(ModuleOp module) {
  TileBarrierReuseLegality verifier;
  return verifier.verify(module);
}

std::unique_ptr<Pass> createTileBarrierReuseLegalityPass() {
  return std::make_unique<TileBarrierReuseLegality>();
}
} // namespace tessera
