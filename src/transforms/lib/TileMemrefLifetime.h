// Shared proof boundary for memref reuse assignment and arena consumption.
#ifndef TESSERA_TRANSFORMS_TILEMEMREFLIFETIME_H
#define TESSERA_TRANSFORMS_TILEMEMREFLIFETIME_H
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>
#include <utility>

namespace tessera::memory {
inline bool isMarker(mlir::Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "tile.alloc_shared" || name == "tile.tmem.alloc";
}
inline int64_t staticBytes(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !type.getElementType().isIntOrFloat()) return -1;
  int64_t bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  for (int64_t dim : type.getShape()) {
    if (dim && bytes > std::numeric_limits<int64_t>::max() / dim) return -1;
    bytes *= dim;
  }
  return bytes;
}
inline mlir::Value viewSource(mlir::Operation *op) {
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(op)) return cast.getSource();
  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) return view.getViewSource();
  return {};
}
inline bool completes(mlir::Operation *wait, mlir::Operation *copy) {
  if (wait->getName().getStringRef() != "tile.wait_async") return false;
  if (wait->getNumOperands()) {
    for (mlir::Value result : copy->getResults())
      if (llvm::is_contained(wait->getOperands(), result)) return true;
    return false;
  }
  for (auto key : {"stage", "tile.barrier_id"})
    if (auto value = wait->getAttr(key))
      if (copy->getAttr(key) != value) return false;
  return true;
}
// Body-derived borrowing summaries. Recursion, external symbols, returned
// aliases, retained descriptors and async accesses are not borrowed calls.
// No user-supplied ownership/noalias attribute can override this proof.
using BorrowSummaries = llvm::DenseMap<std::pair<mlir::Operation *, unsigned>, bool>;
inline bool borrowedArgument(mlir::func::FuncOp fn, unsigned arg,
                             llvm::SmallPtrSetImpl<mlir::Operation *> &active,
                             BorrowSummaries &summaries) {
  if (!fn || fn.isExternal() || !fn.isPrivate() || arg >= fn.getNumArguments()) return false;
  auto key = std::make_pair(fn.getOperation(), arg);
  if (auto cached = summaries.find(key); cached != summaries.end()) return cached->second;
  if (!active.insert(fn).second) return false;
  bool safe = true;
  llvm::SmallPtrSet<mlir::Value, 16> seen;
  llvm::SmallVector<mlir::Value> work{fn.getArgument(arg)};
  while (!work.empty() && safe) {
    auto value = work.pop_back_val();
    if (!seen.insert(value).second) continue;
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (viewSource(user) == value) {
        for (auto result : user->getResults()) work.push_back(result);
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(user)) {
        auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(call, call.getCalleeAttr());
        safe &= borrowedArgument(callee, use.getOperandNumber(), active, summaries);
      } else if (!mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                            mlir::memref::DimOp>(user)) safe = false;
    }
  }
  active.erase(fn);
  summaries[key] = safe;
  return safe;
}
inline bool borrowedCall(mlir::Operation *op, mlir::Value value,
                         bool requireWorkgroupABI = false) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
  if (!call) return false;
  auto fn = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(call, call.getCalleeAttr());
  bool found = false;
  BorrowSummaries summaries;
  for (auto [i, operand] : llvm::enumerate(call.getOperands())) {
    if (operand != value) continue;
    found = true;
    auto type = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
    if (requireWorkgroupABI) {
      auto space = type ? mlir::dyn_cast_or_null<mlir::IntegerAttr>(type.getMemorySpace()) : mlir::IntegerAttr();
      if (!space || space.getInt() != 3) return false;
    }
    llvm::SmallPtrSet<mlir::Operation *, 8> active;
    if (!borrowedArgument(fn, i, active, summaries)) return false;
  }
  return found;
}

inline bool workgroupUniform(mlir::Value value) {
  auto *op = value.getDefiningOp();
  if (!op || !op->isRegistered()) return false;
  auto name = op->getName().getStringRef();
  if (name == "arith.constant" || name == "gpu.block_id" ||
      name == "gpu.grid_dim" || name == "gpu.block_dim") return true;
  if (op->getName().getDialectNamespace() != "arith" || op->getNumRegions()) return false;
  return llvm::all_of(op->getOperands(), workgroupUniform);
}

// Completion must occur on every selected path, including an empty else path.
inline bool completesIn(mlir::Region &region, mlir::Operation *copy, bool sync);
inline bool completionOp(mlir::Operation *op, mlir::Operation *copy, bool sync) {
  auto name = op->getName().getStringRef();
  if (sync ? (name == "tile.cta_sync" || name == "tile.sbarrier") : completes(op, copy)) return true;
  if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
    if (sync && !workgroupUniform(branch.getCondition())) return false;
    llvm::APInt condition;
    if (mlir::matchPattern(branch.getCondition(), mlir::m_ConstantInt(&condition)))
      return completesIn(condition.isZero() ? branch.getElseRegion() : branch.getThenRegion(), copy, sync);
    return completesIn(branch.getThenRegion(), copy, sync) &&
           completesIn(branch.getElseRegion(), copy, sync);
  }
  return false;
}
inline bool completesIn(mlir::Region &region, mlir::Operation *copy, bool sync) {
  if (!llvm::hasSingleElement(region)) return false;
  return llvm::any_of(region.front(), [&](mlir::Operation &op) { return completionOp(&op, copy, sync); });
}

struct Interval {
  int64_t start, end;
  bool reusable;
};
// Local intervals compose with all-path completion and uniform structured
// branch exclusivity. Region-local intervals must finish before their exit,
// including a loop backedge. Escapes and unknown CFG/region forms prevent
// coalescing. Views retain the entire backing allocation conservatively.
class Lifetimes {
  llvm::DenseMap<mlir::Operation *, int64_t> index;
  mlir::Operation *function;
  int64_t end = 0;
public:
  explicit Lifetimes(mlir::Operation *fn) : function(fn) {
    fn->walk([&](mlir::Operation *op) { index[op] = end++; });
  }
  Interval get(mlir::Operation *marker) const {
    Interval live{index.lookup(marker), index.lookup(marker), true};
    auto *block = marker->getBlock();
    auto *owner = block->getParentOp();
    bool localRegion = mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(owner);
    if (!localRegion && (block != &function->getRegion(0).front() ||
        !llvm::hasSingleElement(function->getRegion(0)))) live.reusable = false;
    // All enclosing control decisions must be uniform for collective storage.
    for (auto *parent = owner; parent != function; parent = parent->getParentOp()) {
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent))
        live.reusable &= workgroupUniform(branch.getCondition());
      else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
        live.reusable &= workgroupUniform(loop.getLowerBound()) &&
                         workgroupUniform(loop.getUpperBound()) && workgroupUniform(loop.getStep());
      else live.reusable = false;
    }
    llvm::SmallPtrSet<mlir::Value, 16> seen;
    llvm::SmallVector<mlir::Value> work{marker->getOperand(0)};
    // If the marker is itself a view, account for all sibling aliases too.
    for (mlir::Value root = work.front(); root.getDefiningOp();) {
      auto source = viewSource(root.getDefiningOp());
      if (!source) break;
      work.push_back(source);
      root = source;
    }
    while (!work.empty()) {
      auto value = work.pop_back_val();
      if (!seen.insert(value).second) continue;
      for (mlir::Operation *user : value.getUsers()) {
        if (user == marker) continue;
        live.end = std::max(live.end, index.lookup(user));
        if (user->getBlock() != block || index.lookup(user) < live.start)
          live.reusable = false;
        if (viewSource(user) == value) {
          for (auto result : user->getResults()) work.push_back(result);
          continue;
        }
        auto name = user->getName().getStringRef();
        if (name == "tile.async_copy" || name == "tile.tma.copy_async") {
          int64_t completion = end;
          for (mlir::Operation *next = user->getNextNode(); next; next = next->getNextNode())
            if (completionOp(next, user, false)) { completion = index.lookup(next); break; }
          live.end = std::max(live.end, completion);
        } else if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user) || borrowedCall(user, value)) {
          // Workgroup storage cannot be reassigned until all threads finish
          // synchronous accesses. Program order alone is not a rendezvous.
          int64_t completion = end;
          for (mlir::Operation *next = user->getNextNode(); next; next = next->getNextNode()) {
            if (completionOp(next, nullptr, true)) {
              completion = index.lookup(next); break;
            }
          }
          live.end = std::max(live.end, completion);
        } else if (!isMarker(user) && !mlir::isa<mlir::memref::DimOp>(user)) {
          live.reusable = false;
        }
      }
    }
    // Region-local storage must be released before leaving the region: in a
    // loop this also proves that the next iteration cannot overwrite it.
    if (localRegion && live.end >= index.lookup(block->getTerminator())) live.reusable = false;
    if (!live.reusable) { live.start = 0; live.end = end; }
    return live;
  }
  bool disjoint(mlir::Operation *a, mlir::Operation *b) const {
    if (a->getName() != b->getName() ||
        a->getOperand(0).getType() != b->getOperand(0).getType()) return false;
    auto x = get(a), y = get(b);
    if (!x.reusable || !y.reusable) return false;
    if (a->getBlock() == b->getBlock()) return x.end < y.start || y.end < x.start;
    // Opposite arms of one uniform branch never coexist. Both intervals have
    // independently proved that no access or DMA escapes their arm.
    for (auto *parent = a->getParentOp(); parent && parent != function; parent = parent->getParentOp()) {
      auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent);
      if (!branch || !workgroupUniform(branch.getCondition())) continue;
      auto in = [](mlir::Region &region, mlir::Operation *op) {
        return op->getParentRegion() == &region || region.isAncestor(op->getParentRegion());
      };
      if ((in(branch.getThenRegion(), a) && in(branch.getElseRegion(), b)) ||
          (in(branch.getElseRegion(), a) && in(branch.getThenRegion(), b))) return true;
    }
    return false;
  }
};
} // namespace tessera::memory
#endif
